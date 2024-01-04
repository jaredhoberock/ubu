#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/congruent.hpp"
#include "../concepts/coordinate.hpp"
#include "../detail/as_integral.hpp"
#include "../detail/tuple_algorithm.hpp"

namespace ubu
{

template<scalar_coordinate C1, scalar_coordinate C2>
constexpr bool lexicographical_compare_coordinates(const C1& lhs, const C2& rhs)
{
  return detail::as_integral(lhs) < detail::as_integral(rhs);
}

template<nonscalar_coordinate C1, congruent<C1> C2>
constexpr bool lexicographical_compare_coordinates(const C1& lhs, const C2& rhs)
{
  return detail::tuple_lexicographical_compare(lhs, rhs, [](const auto& l, const auto& r)
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

