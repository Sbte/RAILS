#ifndef TESTHELPERS_H
#define TESTHELPERS_H

#define EXPECT_VECTOR_NEAR(a, b)                        \
    {                                                   \
        int m = (a).M();                                \
        int n = (a).N();                                \
        for (int i = 0; i < m; i++)                     \
            for (int j = 0; j < n; j++)                 \
                EXPECT_NEAR((a)(i,j), (b)(i,j), 1e-3);  \
    }

#define EXPECT_VECTOR_EQ(a, b) {                        \
        int m = (a).M();                                \
        int n = (a).N();                                \
        for (int i = 0; i < m; i++)                     \
            for (int j = 0; j < n; j++)                 \
                EXPECT_DOUBLE_EQ((a)(i,j), (b)(i,j));   \
    }

#define EXPECT_ORTHOGONAL(a) {                          \
        for (int i = 0; i < a.N(); ++i)                 \
        {                                               \
            for (int j = 0; j < a.N(); ++j)             \
            {                                           \
                auto out = a.view(i).dot(a.view(j));    \
                if (i != j)                             \
                    EXPECT_NEAR(0.0, out(0, 0), 1e-15); \
                else                                    \
                    EXPECT_NEAR(1.0, out(0, 0), 1e-15); \
            }                                           \
        }                                               \
    }

#endif
