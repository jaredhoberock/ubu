#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/coordinates/concepts/subdimensional.hpp"
#include "../../../../tensors/views/compose.hpp"
#include "../../../../tensors/views/layouts/extending_layout.hpp"
#include "../../views/memory_view.hpp"
#include "../traits/detail/maybe_allocator_shape.hpp"
#include "custom_allocate.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace ubu::detail
{


// cf. one_extending_default_allocate_after
template<class T, class A, class S>
concept has_one_extending_default_allocate =
  std::is_object_v<T>
  and subdimensional<S,maybe_allocator_shape_t<A>>
  and has_custom_allocate<T,A,maybe_allocator_shape_t<A>>
;


// the purpose of this default path for allocate is to perform simple conversions on the user's arguments (right now, just user_shape)
// to match the allocator's expectations. Then, it forwards the arguments along to the allocator's customization of allocate
//
// the conversion performed on user_shape simply makes it congruent with the allocator's shape type by one-extending the value of user_shape
template<class T, class A, class S>
  requires has_one_extending_default_allocate<T,A&&,S>
constexpr memory_view_of<T,S> auto one_extending_default_allocate(A&& alloc, const S& user_shape)
{
  using AS = maybe_allocator_shape_t<A>;

  // we'll one-extend the user's requested shape and layout the allocator's result tensor using a zero-extending layout
  auto alloc_shape = one_extend_coordinate<AS>(user_shape);

  // allocate a tensor of alloc_shape
  auto view = detail::custom_allocate<T>(std::forward<A>(alloc), alloc_shape);

  // compose with a layout that will zero-extend coordinates from user_shape to alloc_shape
  auto result_view = compose(view, extending_layout<S,AS>(user_shape));

  return result_view;
}


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

