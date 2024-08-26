#pragma once

#include "../../../../detail/prologue.hpp"

#include "../allocate.hpp"
#include "../deallocate.hpp"
#include "../traits/detail/maybe_allocator_shape.hpp"
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ubu
{


// XXX the shape type ought to be a parameter
template<class A, class T>
concept allocator_of =
  std::equality_comparable<A>
  and std::is_nothrow_destructible_v<A>
  and std::is_nothrow_destructible_v<A>

  // we must be able to figure out the element type of tensors allocated by A
  and requires { typename std::remove_cvref_t<A>::value_type; }

  and requires(std::remove_cvref_t<A>& a, detail::maybe_allocator_shape_t<A> shape)
  {
    ubu::deallocate(a, ubu::allocate<T>(a, shape));
  }
;


template<class A>
concept allocator = allocator_of<A, std::byte>;


} // end ubu

#include "../../../../detail/epilogue.hpp"

