#ifndef LYAPUNOVMACROS_H
#define LYAPUNOVMACROS_H

#include "Teuchos_StandardCatchMacros.hpp"

#ifndef CHECK_ZERO
#define CHECK_ZERO(funcall) {                                           \
        int ierr = 0;                                                   \
        bool status = true;                                             \
        try { ierr = funcall; }                                         \
        TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);     \
        if (!status) {                                                  \
            std::string msg = "Caught an exception in call " +          \
                std::string(#funcall) +                                 \
                " on line " + Teuchos::toString(__LINE__) +             \
                " of file " + Teuchos::toString(__FILE__);              \
            std::cerr << msg << std::endl;                              \
            return -1;}                                                 \
        if (ierr) {                                                     \
            std::string msg = "Error code " + Teuchos::toString(ierr) + \
                " returned from call " + std::string(#funcall) +        \
                " on line " + Teuchos::toString(__LINE__) +             \
                " of file " + Teuchos::toString(__FILE__);              \
            std::cerr << msg << std::endl;                              \
            return ierr;}                                               \
    }
#endif

#endif
