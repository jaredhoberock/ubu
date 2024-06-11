#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../miscellaneous/integrals/detail/as_integral.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include <concepts>

namespace ubu
{
namespace detail
{

template<coordinate C>
constexpr congruent<C> auto convert_non_integral_elements(const C& coord)
{
  if constexpr (integral_like<C>)
  {
    return as_integral(coord);
  }
  else
  {
    return tuple_zip_with(coord, [](const auto& c_i)
    {
      return convert_non_integral_elements(c_i);
    });
  }
}

} // end detail


// Given some shape S, we need some default type of
// coordinate we should use for a tensor with that shape.
//
// Ordinarily, the coordinate type and shape type would be the same.
//
// The issue is that some types of shape are constant (i.e., ubu::constant)
// and cannot be used as coordinates into a tensor because such types 
// only take on a single constant value.
//
// To deal with this issue, we use the type returned by
// convert_non_integral_elements above, which converts any constant parts of
// S into the corresponding std::integral type which has a dynamic value.
template<coordinate S>
using default_coordinate_t = decltype(detail::convert_non_integral_elements(std::declval<S>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

