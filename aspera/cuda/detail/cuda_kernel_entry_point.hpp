#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>
#include <type_traits>

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<std::invocable F>
  requires std::is_trivially_copyable_v<F>
__global__ void cuda_kernel_entry_point(F f)
{
  f();
}


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

