#pragma once

#include "../../../detail/prologue.hpp"

#include "../congruent.hpp"
#include "../coordinate.hpp"
#include "tuple_algorithm.hpp"
#include <tuple>
#include <utility>

namespace ubu::detail
{


// returns (quotient, remainder)
// quotient and remainder are always std::integral
template<scalar_coordinate C1, scalar_coordinate C2>
constexpr auto coordinate_divmod(const C1& dividend, const C2& divisor)
{
  auto quotient = element<0>(dividend) / element<0>(divisor);
  auto remainder = element<0>(dividend) % element<0>(divisor);

  return std::pair(quotient, remainder);
}


// returns (quotient, remainder)
// the returned remainder is congruent with C2
template<scalar_coordinate C1, nonscalar_coordinate C2>
constexpr coordinate auto coordinate_divmod(const C1& dividend, const C2& divisor)
{
  return tuple_fold(std::pair(dividend, std::tuple()), divisor, [](auto prev, auto current_divisor)
  {
    auto [prev_quotient, prev_remainder] = prev;
    auto [quotient, remainder] = coordinate_divmod(prev_quotient, current_divisor);

    // ensure that the tuple type of the remainder is similar to what we started with in C2
    return std::pair(quotient, tuple_append_similar_to<C2>(prev_remainder, remainder));
  });
}


// returns (quotient, remainder)
// the returned remainder is congruent with C2
template<nonscalar_coordinate C1, nonscalar_coordinate C2>
  requires weakly_congruent<C1,C2>
constexpr coordinate auto coordinate_divmod(const C1& dividend, const C2& divisor)
{
  return tuple_unzip(tuple_zip_with(dividend, divisor, [](const auto& dividend, const auto& divisor)
  {
    return coordinate_divmod(dividend, divisor);
  }));
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

