// note that this header file is special and does not use #pragma once

// ASPERA_HAS_EXCEPTIONS indicates whether or not exception support is available.

#ifndef ASPERA_HAS_EXCEPTIONS

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__)
#      define ASPERA_HAS_EXCEPTIONS __cpp_exceptions
#    else
#      define ASPERA_HAS_EXCEPTIONS 0
#    endif
#  else
#    define ASPERA_HAS_EXCEPTIONS __cpp_exceptions
#  endif

#elif defined(ASPERA_HAS_EXCEPTIONS)
#  undef ASPERA_HAS_EXCEPTIONS
#endif

