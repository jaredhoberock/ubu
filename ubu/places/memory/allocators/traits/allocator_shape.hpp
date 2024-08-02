#pragma once

#include "../../../../detail/prologue.hpp"

#include "../concepts/asynchronous_allocator.hpp"
#include "allocator_size.hpp"
#include <type_traits>

namespace ubu
{
namespace detail
{

template<class A>
struct allocator_shape
{
  using type = allocator_size_t<A>;
};

template<class A>
  requires requires { typename std::remove_cvref_t<A>::shape_type; }
struct allocator_shape<A>
{
  using type = typename std::remove_cvref_t<A>::shape_type;
};

} // end detail

template<asynchronous_allocator A>
using allocator_shape_t = typename detail::allocator_shape<A>::type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

