#include "Epetra_OperatorWrapper.hpp"
#include "Epetra_MultiVectorWrapper.hpp"
#include "Epetra_SerialDenseMatrixWrapper.hpp"

#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>
#include <Epetra_CrsMatrix.h>

#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziEpetraAdapter.hpp>
#include <AnasaziBlockKrylovSchurSolMgr.hpp>

#define TIMER_ON
#include "Timer.hpp"

Epetra_OperatorWrapper::Epetra_OperatorWrapper()
    :
    ptr_(Teuchos::null)
{}

Epetra_OperatorWrapper::Epetra_OperatorWrapper(Teuchos::RCP<Epetra_Operator> ptr)
    :
    Epetra_OperatorWrapper()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "constructor 1");
    ptr_ = ptr;
}

Epetra_OperatorWrapper::Epetra_OperatorWrapper(Epetra_OperatorWrapper const &other)
    :
    Epetra_OperatorWrapper()
{
    FUNCTION_TIMER("Epetra_OperatorWrapper", "constructor 2");
    ptr_ = other.ptr_;
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
    Epetra_MultiVectorWrapper out(Teuchos::rcp(new Epetra_MultiVector(*other)));
    ptr_->Apply(*other, *out);
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
