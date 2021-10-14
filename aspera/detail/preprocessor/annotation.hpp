// note that this header file is special and does not use #pragma once

// ASPERA_ANNOTATION expands to __host__ __device__ when encountered by a
// CUDA-capable compiler

#if !defined(ASPERA_ANNOTATION)

#  ifdef __CUDACC__
#    define ASPERA_ANNOTATION __host__ __device__
#  else
#    define ASPERA_ANNOTATION
#  endif
#  define ASPERA_ANNOTATION_NEEDS_UNDEF

#elif defined(ASPERA_ANNOTATION_NEEDS_UNDEF)

#undef ASPERA_ANNOTATION
#undef ASPERA_ANNOTATION_NEEDS_UNDEF

#endif

