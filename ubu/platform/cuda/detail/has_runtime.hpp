#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../detail/reflection/is_host.hpp"


namespace ubu::cuda::detail
{


constexpr bool has_runtime()
{
  bool result = false;

#if defined(__circle_lang__)
  if target(ubu::detail::is_host())
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


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

