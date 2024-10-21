#pragma once

#include "../../../detail/prologue.hpp"

#include <concepts>
#include <type_traits>

namespace ubu::cuda::detail
{


#if defined(__CUDACC__)


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
__global__ void kernel_entry_point(F f)
{
  f();
}


#endif


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
using kernel_entry_point_t = void(*)(F);


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
kernel_entry_point_t<F> kernel_for_invocable(F f)
{
#if defined(__CUDACC__)
  return &kernel_entry_point<F>;
#else
  void* null_void_ptr = nullptr; 
  return reinterpret_cast<kernel_entry_point_t<F>>(null_void_ptr);
#endif
}


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

