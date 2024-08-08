#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/strictly_subdimensional.hpp"
#include "../../../../tensors/views/compose.hpp"
#include "../../../../tensors/views/layouts/extending_layout.hpp"
#include "../../../causality/asynchronous_view_of.hpp"
#include "../../../causality/happening.hpp"
#include "../concepts/allocator.hpp"
#include "../traits/allocator_shape.hpp"
#include "custom_allocate_after.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace ubu::detail
{

// cf. has_one_extending_bulk_execute_after
template<class T, class A, class B, class S>
concept has_one_extending_default_allocate_after =
  std::is_object_v<T>
  and allocator<A>
  and happening<B>
  and strictly_subdimensional<S,allocator_shape_t<A>>
  and has_custom_allocate_after<T,A,B,allocator_shape_t<A>>
;


// cf. one_extending_bulk_execute_after
//
// the purpose of this default path for allocate_after is to perform simple conversions on the user's arguments (right now, just user_shape)
// to match the allocator's expectations. Then, it simply forwards the arguments along to the allocator's customization of allocate_after
//
// the conversion performed on user_shape simply makes it congruent with allocator_shape_t<A> by one-extending the value of user_shape
//
// in principle, we could also convert the before argument into the allocator's happening type, if the type of before doesn't match allocator_happening_t<A>
template<class T, allocator A, happening B, strictly_subdimensional<allocator_shape_t<A>> S>
  requires has_one_extending_default_allocate_after<T,A&&,B&&,S>
constexpr asynchronous_view_of<T,S> auto one_extending_default_allocate_after(A&& alloc, B&& before, const S& user_shape)
{
  // we'll one-extend the user's requested shape and layout the allocator's result tensor using a zero-extending layout
  auto alloc_shape = one_extend_coordinate<allocator_shape_t<A>>(user_shape);

  // allocate a tensor of alloc_shape
  auto [after, view] = detail::custom_allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), alloc_shape);

  // compose with a layout that will zero-extend coordinates from user_shape to alloc_shape
  auto result_view = compose(view, extending_layout<S,allocator_shape_t<A>>(user_shape));

  return std::pair(std::move(after), result_view);
}

} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

