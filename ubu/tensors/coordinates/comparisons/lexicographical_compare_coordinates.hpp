#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/tuples.hpp"
#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/to_integral_like.hpp"

namespace ubu
{

template<scalar_coordinate C1, scalar_coordinate C2>
constexpr bool lexicographical_compare_coordinates(const C1& lhs, const C2& rhs)
{
  return detail::to_integral_like(lhs) < detail::to_integral_like(rhs);
}

template<nonscalar_coordinate C1, congruent<C1> C2>
constexpr bool lexicographical_compare_coordinates(const C1& lhs, const C2& rhs)
{
  return tuples::lexicographical_compare(lhs, rhs, [](const auto& l, const auto& r)
  {
    return lexicographical_compare_coordinates(l,r);
  });
}

namespace detail
{

struct lex_less_t
{
  template<coordinate C1, congruent<C1> C2>
  constexpr bool operator()(const C1& lhs, const C2& rhs) const
  {
    return lexicographical_compare_coordinates(lhs, rhs);
  }
};

} // end detail

constexpr detail::lex_less_t lex_less{};


} // end ubu

#include "../../../detail/epilogue.hpp"

