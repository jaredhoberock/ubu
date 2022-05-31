#pragma once

#include "../prologue.hpp"

#include "is_host.hpp"

UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


constexpr bool has_cuda_runtime()
{
  bool result = false;


#if defined(__circle_lang__)
  if target(is_host())
  {
    // XXX we could consider using #__has_include(<cuda_runtime_api.h>) here
    result = true;
  }
  else if target(__nvvm_arch >= nvvm_arch_t{35})
  {
#  if defined(__CUDACC_RDC__)
    // in __device__ code, using the CUDA Runtime requires relocatable device code
    result = true;
#  endif // __CUDACC_RDC__
  }
#else // __circle_lang__
  result = true;
#endif

  return result;
}


} // end detail

UBU_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

