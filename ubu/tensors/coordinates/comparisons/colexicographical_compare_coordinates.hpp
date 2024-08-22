#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"

namespace ubu
{

template<coordinate L, congruent<L> R>
constexpr bool colexicographical_compare_coordinates(const L& lhs, const R& rhs)
{
  if constexpr (unary_coordinate<L>)
  {
    return detail::to_integral_like(lhs) < detail::to_integral_like(rhs);
  }
  else
  {
    return tuples::colexicographical_compare(lhs, rhs, [](const auto& l, const auto& r)
    {
      return colexicographical_compare_coordinates(l,r);
    });
  }
}


namespace detail
{

struct colex_less_t
{
  template<coordinate L, congruent<L> R>
  constexpr bool operator()(const L& lhs, const R& rhs) const
  {
    return colexicographical_compare_coordinates(lhs, rhs);
  }
};

} // end detail

inline constexpr detail::colex_less_t colex_less;


} // end ubu

#include "../../../detail/epilogue.hpp"

