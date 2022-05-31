#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>
#include <type_traits>

UBU_NAMESPACE_OPEN_BRACE


namespace detail
{


#if defined(__CUDACC__)


template<std::invocable F>
  requires std::is_trivially_copyable_v<F>
__global__ void cuda_kernel_entry_point(F f)
{
  f();
}


#endif


} // end detail


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

