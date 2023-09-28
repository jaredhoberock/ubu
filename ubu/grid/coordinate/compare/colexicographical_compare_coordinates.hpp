#pragma once

#include "../../../detail/prologue.hpp"

#include "../congruent.hpp"
#include "../coordinate.hpp"
#include "../detail/tuple_algorithm.hpp"
#include "../element.hpp"

namespace ubu
{

template<scalar_coordinate C1, scalar_coordinate C2>
constexpr bool colexicographical_compare_coordinates(const C1& lhs, const C2& rhs)
{
  return element<0>(lhs) < element<0>(rhs);
}

template<nonscalar_coordinate C1, congruent<C1> C2>
constexpr bool colexicographical_compare_coordinates(const C1& lhs, const C2& rhs)
{
  return detail::tuple_colexicographical_compare(lhs, rhs, [](const auto& l, const auto& r)
  {
    return colexicographical_compare_coordinates(l,r);
  });
}


namespace detail
{

struct colex_less_t
{
  template<coordinate C1, congruent<C1> C2>
  constexpr bool operator()(const C1& lhs, const C2& rhs) const
  {
    return colexicographical_compare_coordinates(lhs, rhs);
  }
};

} // end detail

constexpr detail::colex_less_t colex_less{};


} // end ubu

#include "../../../detail/epilogue.hpp"

