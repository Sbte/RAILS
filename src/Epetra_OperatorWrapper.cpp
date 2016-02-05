#include "Epetra_OperatorWrapper.hpp"
#include "Epetra_MultiVectorWrapper.hpp"
#include "Epetra_SerialDenseMatrixWrapper.hpp"

#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>
#include <Epetra_CrsMatrix.h>

#include <Epetra_SerialComm.h>
#include <Epetra_Map.h>

#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziEpetraAdapter.hpp>
#include <AnasaziBlockKrylovSchurSolMgr.hpp>

#define TIMER_ON
#include "Timer.hpp"

Epetra_OperatorWrapper::Epetra_OperatorWrapper()
    :
    ptr_(Teuchos::null),
    transpose_(false)
{}

Epetra_OperatorWrapper::Epetra_OperatorWrapper(Teuchos::RCP<Epetra_Operator> ptr)
    :
    Epetra_OperatorWrapper()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "constructor 1");
    ptr_ = ptr;

    if (!ptr.is_null())
        transpose_ = ptr->UseTranspose();
}

Epetra_OperatorWrapper::Epetra_OperatorWrapper(Epetra_OperatorWrapper const &other)
    :
    Epetra_OperatorWrapper()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "constructor 2");
    ptr_ = other.ptr_;
    transpose_ = other.transpose_;
}

Epetra_OperatorWrapper Epetra_OperatorWrapper::transpose() const
{
    Epetra_OperatorWrapper tmp(*this);
    tmp.transpose_ = !tmp.transpose_;
    return tmp;
}

int Epetra_OperatorWrapper::set_parameters(Teuchos::ParameterList &params)
{
    params_ = Teuchos::rcp(&params, false);
    return 0;
}

Epetra_Operator &Epetra_OperatorWrapper::operator *()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "*");
    return *ptr_;
}

Epetra_Operator const &Epetra_OperatorWrapper::operator *() const
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "* 2");
    return *ptr_;
}

Epetra_MultiVectorWrapper Epetra_OperatorWrapper::operator *(
    Epetra_MultiVectorWrapper const &other) const
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "* MV");

    bool use_transpose = ptr_->UseTranspose();
    ptr_->SetUseTranspose(transpose_);

    // Epetra_MultiVectorWrapper out(Teuchos::rcp(new Epetra_MultiVector(*other)));
    Epetra_MultiVectorWrapper out(
        Teuchos::rcp(new Epetra_MultiVector(
                         ptr_->OperatorRangeMap(), other.N())));
    ptr_->Apply(*other, *out);

    ptr_->SetUseTranspose(use_transpose);
    return out;
}

double Epetra_OperatorWrapper::norm(int n)
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "norm");
    Teuchos::RCP<Epetra_CrsMatrix> mat =
        Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(ptr_);
    if (!mat.is_null())
        return mat->NormFrobenius();
    return ptr_->NormInf();
}

int Epetra_OperatorWrapper::eigs(Epetra_MultiVectorWrapper &V,
                                 Epetra_SerialDenseMatrixWrapper &D,
                                 int num, double tol) const
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "eigs");
    Teuchos::RCP<Teuchos::ParameterList> params;
    if (params_.is_null())
        params = Teuchos::rcp(new Teuchos::ParameterList);
    else
        params = params_;

    Teuchos::ParameterList &eig_params = params->sublist("Eigenvalue Solver");

    if (!eig_params.isParameter("Convergence Tolerance"))
        eig_params.set("Convergence Tolerance", tol);

    tol = eig_params.get("Convergence Tolerance", tol);

    //TODO: Maybe stop here if the eigenvalues become too small?

    Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(
        new Epetra_Vector(ptr_->OperatorDomainMap()));
    x->PutScalar(1.0);

    Teuchos::RCP<Anasazi::BasicEigenproblem<
        double, Epetra_MultiVector, Epetra_Operator> > eig_problem =
        Teuchos::rcp(new Anasazi::BasicEigenproblem<
                     double, Epetra_MultiVector, Epetra_Operator>(ptr_, x));
    eig_problem->setHermitian(true);
    eig_problem->setNEV(num);

    eig_problem->setProblem();

    Anasazi::BlockKrylovSchurSolMgr<
        double, Epetra_MultiVector, Epetra_Operator>
        sol_manager(eig_problem, eig_params);

    Anasazi::ReturnType ret;
    ret = sol_manager.solve();
    if (ret != Anasazi::Converged)
    {
        std::cerr << "Eigensolver did not converge" << std::endl;
        return ret;
    }

    const Anasazi::Eigensolution<double, Epetra_MultiVector> &eig_sol =
        eig_problem->getSolution();

    const std::vector<Anasazi::Value<double> > &evals = eig_sol.Evals;
    int num_eigs = evals.size();
    D.resize(num_eigs, 1);

    for (int i = 0; i < num_eigs; i++)
    {
        // TODO: We want Anasazi to detect this so we can stop earlier.
        if (std::abs(evals[i].realpart) < tol)
        {
            num_eigs = i;
            break;
        }

        D(i, 0) = evals[i].realpart;
    }

    D.resize(num_eigs, 1);

    V = eig_sol.Evecs;
    V.resize(num_eigs);

    return 0;
}

Epetra_OperatorWrapper::Epetra_OperatorWrapper(int m, int n)
    :
    Epetra_OperatorWrapper()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "constructor 3");

    Epetra_SerialComm comm;
    Epetra_Map map(m, 0, comm);
    Teuchos::RCP<Epetra_CrsMatrix> mat =
        Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, n));

    double *values = new double[n]();
    int *indices = new int[n]();
    for (int i = 0; i < m; ++i)
        indices[i] = i;

    for (int i = 0; i < m; ++i)
        mat->InsertGlobalValues(i, n, values, indices);

    mat->FillComplete();

    delete[] values;
    delete[] indices;

    ptr_ = mat;
}

double &Epetra_OperatorWrapper::operator ()(int m, int n)
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "()");

    double *values;
    int num_entries;

    Teuchos::RCP<Epetra_CrsMatrix> mat =
        Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(ptr_);
    if (mat.is_null())
        std::cerr << "Matrix is not a CrsMatrix" << std::endl;
    
    mat->ExtractMyRowView(m, num_entries, values);
    return values[n];
}

double const &Epetra_OperatorWrapper::operator ()(int m, int n) const
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "() 2");

    double *values;
    int num_entries;

    Teuchos::RCP<Epetra_CrsMatrix> mat =
        Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(ptr_);
    if (mat.is_null())
        std::cerr << "Matrix is not a CrsMatrix" << std::endl;
    
    mat->ExtractMyRowView(m, num_entries, values);
    return values[n];
}
