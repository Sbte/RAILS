#ifndef MATRIXORMULTIVECTORWRAPPER_H
#define MATRIXORMULTIVECTORWRAPPER_H

template<class Matrix, class MultiVector>
class MatrixOrMultiVectorWrapper
{
    bool is_matrix_;
    Matrix matrix_;
    MultiVector vector_;
    bool transpose_;
public:
    MatrixOrMultiVectorWrapper() = delete;

    MatrixOrMultiVectorWrapper(Matrix const &other)
        :
        is_matrix_(true),
        matrix_(other),
        transpose_(false)
        {}

    MatrixOrMultiVectorWrapper(MultiVector const &other)
        :
        is_matrix_(false),
        vector_(other),
        transpose_(false)
        {}

    virtual ~MatrixOrMultiVectorWrapper() {}

    double norm() const
        {
            if (is_matrix_)
                return matrix_.norm();
            return vector_.norm();
        }

    MatrixOrMultiVectorWrapper transpose() const
        {
            MatrixOrMultiVectorWrapper tmp(*this);
            tmp.transpose_ = !tmp.transpose_;
            return tmp;
        }

    MultiVector operator *(MultiVector const &other) const
        {
            if (transpose_)
            {
                if (is_matrix_)
                    return std::move(matrix_.transpose() * other);
                else
                    return std::move(vector_.transpose() * other);
            }
            if (is_matrix_)
                return std::move(matrix_ * other);
            else
                return std::move(vector_ * other);
        }
};

template<class Type>
class MatrixOrMultiVectorWrapper<Type, Type>
{
    Type type_;
    bool transpose_;
public:
    MatrixOrMultiVectorWrapper() = delete;

    template<class MatrixOrMultiVector>
    MatrixOrMultiVectorWrapper(MatrixOrMultiVector const &other)
        :
        type_(other),
        transpose_(false)
        {}

    virtual ~MatrixOrMultiVectorWrapper() {}

    double norm() const
        {
            return type_.norm();
        }

    MatrixOrMultiVectorWrapper transpose() const
        {
            MatrixOrMultiVectorWrapper tmp(*this);
            tmp.transpose_ = !tmp.transpose_;
            return tmp;
        }

    Type operator *(Type const &other) const
        {
            if (transpose_)
                return std::move(type_.transpose() * other);
            return std::move(type_ * other);
        }
};

#endif
