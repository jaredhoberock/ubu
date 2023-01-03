#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/coordinate.hpp"
#include "../../coordinate/detail/tuple_algorithm.hpp"
#include <tuple>


namespace ubu::detail
{

// in the following, divisor is a binary function (dividend:int, divisor:int) -> (quotient:int, remainder:int)


// returns the result of divider(dividend, divisor) as the pair (quotient, remainder)
template<scalar_coordinate C1, scalar_coordinate C2, class D>
constexpr pair_like auto divide_coordinate(const C1& dividend, const C2& divisor, D divider)
{
  return divider(element<0>(dividend), element<0>(divisor));
}


// XXX this one is the form weakly_congruent<C2,C1>
template<nonscalar_coordinate C1, scalar_coordinate C2, class D>
constexpr pair_like auto divide_coordinate(const C1& dividend, const C2& divisor, D divider)
{
  using namespace std;

  return tuple_fold(pair(tuple(), divisor), dividend, [divider](const auto& prev, const auto& current_dividend)
  {
    auto [prev_quotient, current_divisor] = prev;
    auto [current_quotient, current_remainder] = divide_coordinate(current_dividend, current_divisor, divider);

    return pair(tuple_append_similar_to<C1>(prev_quotient, current_quotient), current_remainder);
  });
}


} // end ubu::detail


#include "../../../detail/epilogue.hpp"

