#include <limits.h>
#include "gtest/gtest.h"

#include "StlWrapper.hpp"



template <class MultiVector>
class GenericMultiVectorWrapperTest: public testing::Test {
protected:
    GenericMultiVectorWrapperTest()
        :
        a_(),
        b_(),
        c_()
        {}

    virtual ~GenericMultiVectorWrapperTest() {}

    MultiVector a_;
    MultiVector b_;
    MultiVector c_;
};

#if GTEST_HAS_TYPED_TEST

using testing::Types;

typedef Types<StlWrapper> Implementations;

TYPED_TEST_CASE(GenericMultiVectorWrapperTest, Implementations);

TYPED_TEST(GenericMultiVectorWrapperTest, ReturnsFalseForNonPrimes)
{
    
