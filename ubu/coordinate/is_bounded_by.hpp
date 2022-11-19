#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "coordinate.hpp"
#include "detail/tuple_algorithm.hpp"
#include "element.hpp"


namespace ubu
{


// [origin, end) defines a half-open interval
template<scalar_coordinate C, scalar_coordinate O, scalar_coordinate E>
constexpr bool is_bounded_by(const C& coord, const O& origin, const E& end)
{
  return element<0>(origin) <= element<0>(coord) and element<0>(coord) < element<0>(end);
}

// (origin, end) defines a bounding box whose axes are half-open intervals
// i.e., origin is included in the box, end is not included in the box
template<nonscalar_coordinate C, nonscalar_coordinate O, nonscalar_coordinate E>
  requires congruent<C,O,E>
constexpr bool is_bounded_by(const C& coord, const O& origin, const E& end)
{
  auto tuple_of_results = detail::tuple_zip_with(coord, origin, end, [](auto c, auto o, auto e)
  {
    return is_bounded_by(c, o, e);
  });

  return detail::tuple_all(tuple_of_results, [](bool result)
  {
    return result;
  });
}


} // end ubu

#include "../detail/epilogue.hpp"

