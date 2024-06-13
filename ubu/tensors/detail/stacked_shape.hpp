#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinates.hpp"
#include "../coordinates/detail/tuple_algorithm.hpp"
#include "as_tuple_of_constants.hpp"
#include "coordinate_cat.hpp"
#include "coordinate_max.hpp"

namespace ubu::detail
{

// this returns the shape of the tensor created by stacking a tensor with shape_a and a tensor with shape_b along axis
template<int axis, coordinate A, congruent<A> B>
  requires (axis <= rank_v<A>)
constexpr coordinate auto stacked_shape(const A& shape_a, const B& shape_b)
{
  if constexpr (axis == rank_v<A>)
  {
    return coordinate_cat(coordinate_max(shape_a,shape_b),2);
  }
  else if constexpr (scalar_coordinate<A>)
  {
    return coordinate_sum(shape_a, shape_b);
  }
  else
  {
    // auto axes = tuple(constant<I>()...);
    constexpr auto axes = as_tuple_of_constants(tuple_indices<A>);

    return tuple_zip_with(shape_a, shape_b, axes, [](const auto& a, const auto& b, auto current_axis)
    {
      if constexpr (current_axis == axis)
      {
        return coordinate_sum(a, b);
      }
      else
      {
        return coordinate_max(a, b);
      }
    });
  }
}

} // end ubu::detail

#include "../../detail/epilogue.hpp"

