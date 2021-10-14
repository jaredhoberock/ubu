// note that this header file is special and does not use #pragma once

// ASPERA_EXEC_CHECK_DISABLE expands to a pragma which tells a CUDA-capable
// compiler not to enforce that a function must call another function with
// a compatible execution space

#ifndef ASPERA_EXEC_CHECK_DISABLE
#  ifdef __CUDACC__
#    define ASPERA_EXEC_CHECK_DISABLE #pragma nv_exec_check_disable
#  else
#    define ASPERA_EXEC_CHECK_DISABLE
#  endif
#elif defined(ASPERA_EXEC_CHECK_DISABLE)
#  undef ASPERA_EXEC_CHECK_DISABLE
#endif

