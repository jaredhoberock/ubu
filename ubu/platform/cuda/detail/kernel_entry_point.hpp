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


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

