#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/coordinate.hpp"
#include "../allocate.hpp"
#include "../deallocate.hpp"
#include "../traits/detail/maybe_allocator_shape.hpp"
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ubu
{


template<class A, class T, class S = detail::maybe_allocator_shape_t<A>>
concept allocator_of =
  std::equality_comparable<A>
  and std::is_nothrow_destructible_v<A>
  and std::is_nothrow_destructible_v<A>

  and std::is_object_v<T>
  and coordinate<S>

  // we must be able to figure out the element type of tensors allocated by A
  and requires { typename std::remove_cvref_t<A>::value_type; }

  and requires(std::remove_cvref_t<A>& a, S shape)
  {
    ubu::deallocate(a, ubu::allocate<T>(a, shape));
  }
;


template<class A>
concept allocator = allocator_of<A, std::byte, detail::maybe_allocator_shape_t<A>>;


} // end ubu

#include "../../../../detail/epilogue.hpp"

