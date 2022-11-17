#pragma once

#include "../detail/prologue.hpp"

#include "congruent.hpp"
#include "coordinate.hpp"
#include "detail/tuple_algorithm.hpp"
#include "element.hpp"
#include <tuple>
#include <utility>


namespace ubu
{


// returns (empty tuple, dividend)
// XXX is this actually right? how come this operation is a no-op for nonscalar coordinates
//     but a division for scalar coordinates?
template<nonscalar_coordinate C, nonscalar_coordinate S>
  requires congruent<C,S>
constexpr coordinate auto colexicographic_lift(const C& dividend, const S&)
{
  return std::pair{std::make_tuple(), dividend};
}


// returns (quotient, remainder)
// the returned remainder is congruent with C2
template<scalar_coordinate C1, scalar_coordinate C2>
constexpr coordinate auto colexicographic_lift(const C1& dividend, const C2& divisor)
{
  auto quotient  = element<0>(dividend) / element<0>(divisor);
  auto remainder = element<0>(dividend) % element<0>(divisor);

  return std::pair{quotient, remainder};
}


// returns (quotient, remainder)
// the returned remainder is congruent with C2
template<scalar_coordinate C1, nonscalar_coordinate C2>
constexpr coordinate auto colexicographic_lift(const C1& dividend, const C2& divisor)
{
  return detail::tuple_fold(std::make_pair(dividend, std::make_tuple()), divisor, [](auto prev, auto s)
  {
    auto [prev_quotient, prev_remainder] = prev;
    auto [quotient, remainder] = colexicographic_lift(prev_quotient, s);

    // ensure that the tuple type of the remainder is similar to what we started with in C2
    return std::pair{quotient, detail::tuple_append_similar_to<C2>(prev_remainder, remainder)};
  });
}


// returns (quotient, remainder)
// the returned remainder is congruent with C2
template<nonscalar_coordinate C1, nonscalar_coordinate C2>
  requires weakly_congruent<C1,C2>
constexpr coordinate auto colexicographic_lift(const C1& dividend, const C2& divisor)
{
  return detail::tuple_unzip(detail::tuple_zip_with(dividend, divisor, [](const auto& dividend, const auto& divisor)
  {
    return colexicographic_lift(dividend, divisor);
  }));
}


} // end ubu

#include "../detail/epilogue.hpp"

