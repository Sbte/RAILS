#ifndef EPETRA_OPERATORWRAPPER_H
#define EPETRA_OPERATORWRAPPER_H

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_MultiVectorWrapper.hpp"

class Epetra_Operator;
class Epetra_BlockMap;
class Epetra_SerialDenseMatrixWrapper;

template<class Operator>
class OperatorFromApplyMethod;

class Epetra_OperatorWrapper
{
    Teuchos::RCP<Epetra_Operator> ptr_;
    Teuchos::RCP<Teuchos::ParameterList> params_;

    bool transpose_;
public:
    Epetra_OperatorWrapper();
    Epetra_OperatorWrapper(Teuchos::RCP<Epetra_Operator> ptr);
    Epetra_OperatorWrapper(Epetra_OperatorWrapper const &other);

    template<class Operator>
    static Epetra_OperatorWrapper from_operator(Operator &op)
        {
            return Epetra_OperatorWrapper(
                Teuchos::rcp(new OperatorFromApplyMethod<Operator>(op)));
        }

    Epetra_OperatorWrapper transpose() const;

    virtual ~Epetra_OperatorWrapper() {}

    Epetra_Operator &operator *();
    Epetra_Operator const &operator *() const;

    Epetra_MultiVectorWrapper operator *(Epetra_MultiVectorWrapper const &other) const;

    int set_parameters(Teuchos::ParameterList &params);

    int M() const;
    int N() const;

    double norm(int n = 0);
    int eigs(Epetra_MultiVectorWrapper &V,
             Epetra_SerialDenseMatrixWrapper &D,
             int num, double tol = 1e-6) const;

// Test methods that do not have to be exposed
protected:
    Epetra_OperatorWrapper(int m, int n);

    double &operator ()(int m, int n = 0);
    double const &operator ()(int m, int n = 0) const;
};

#include <Epetra_Operator.h>
#include <Epetra_Map.h>

template<class Operator>
class OperatorFromApplyMethod: public Epetra_Operator
{
    Operator &op_;
    mutable Teuchos::RCP<Epetra_Map> map_;
public:
    OperatorFromApplyMethod(Operator &op): op_(op) {}

    virtual ~OperatorFromApplyMethod() {};

    int SetUseTranspose(bool UseTranspose)
        { return -1;};

    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
        {
            Epetra_MultiVectorWrapper XW(
                Teuchos::rcp_const_cast<Epetra_MultiVector>(
                    Teuchos::rcp(&X, false)));
            Epetra_MultiVectorWrapper YW(
                Teuchos::rcp(&Y, false));
            YW.view() = op_ * XW;
            return 0;
        }

    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
        { return -1;}

    double NormInf() const
        { return 0.0;}

    const char * Label() const
        { return "";}

    bool UseTranspose() const
        { return false;}

    bool HasNormInf() const
        { return false;}

    const Epetra_Comm & Comm() const
        { return (*op_.V).Comm();}

    const Epetra_Map & OperatorDomainMap() const
        {
            ConstructMap();
            return *map_;
        }

    const Epetra_Map & OperatorRangeMap() const
        {
            ConstructMap();
            return *map_;
        }

    int ConstructMap() const
        {
            if (!map_.is_null())
                return 0;

            Epetra_BlockMap const &map = (*op_.V).Map();
            if (!map.ConstantElementSize() || map.MaxElementSize() != 1)
            {
                std::cerr << "BlockMap is not a Map" << std::endl;
                return -1;
            }

            map_ = Teuchos::rcp(new Epetra_Map(map.NumGlobalElements(),
                                               map.IndexBase(), map.Comm()));
            return 0;
        }
};

#endif
