// note that this header file is special and does not use #pragma once

// ASPERA_HAS_CUDART indicates whether or not the CUDA Runtime API is available.

#ifndef ASPERA_HAS_CUDART

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#      define ASPERA_HAS_CUDART 1
#    else
#      define ASPERA_HAS_CUDART 0
#    endif
#  else
#    define ASPERA_HAS_CUDART 0
#  endif

#elif defined(ASPERA_HAS_CUDART)
#  undef ASPERA_HAS_CUDART
#endif

