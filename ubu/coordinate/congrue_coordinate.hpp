#pragma once

#include "coordinate.hpp"
#include "detail/coordinate_divmod.hpp"
#include "detail/tuple_algorithm.hpp"
#include <tuple>
#include <utility>


namespace ubu
{


template<coordinate C1, coordinate C2>
  requires congruent<C1,C2>
constexpr C1 congrue_coordinate(const C1& coord, const C2&)
{
  return coord;
}


template<scalar_coordinate C1, nonscalar_coordinate C2>
constexpr congruent<C2> auto congrue_coordinate(const C1& coord, const C2& shape)
{
  // XXX the use of unwrap/wrap here deals with a case like the following:
  // shape = ((a,b,c), (d,e))
  //
  // Dropping the last element of shape yields a degenerate
  // shape_front = ((a,b,c)) with superfluous parentheses
  // which is not currently recognized as a coordinate
  //
  // so we unwrap single-element shape_front
  // and then apply a wrapping to remainder to account for it

  // to avoid periodic behavior on the last mode of this operation,
  // split the shape at its last element
  // use divmod on the front of the shape and then concatenate that result
  // with congrue_coordinate applied to the last element

  auto shape_front = detail::tuple_unwrap_single(detail::tuple_drop_last(shape));
  auto shape_last  = detail::tuple_last(shape);

  auto [quotient, remainder] = detail::coordinate_divmod(coord, shape_front);

  auto final_mode = congrue_coordinate(quotient, shape_last);

  return detail::tuple_append_similar_to<C2>(detail::tuple_wrap_if<rank_v<C2> == 2>(remainder), final_mode);
}


// XXX is this implementation correct?
template<nonscalar_coordinate C1, nonscalar_coordinate C2>
  requires (not congruent<C1,C2> and weakly_congruent<C1,C2>)
constexpr congruent<C2> auto congrue_coordinate(const C1& coord, const C2& shape)
{
  return detail::tuple_zip_with(coord, shape, [](const auto& c, const auto& s)
  {
    return congrue_coordinate(c, s);
  });
}


} // end ubu

