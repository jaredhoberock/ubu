#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/coordinate.hpp"
#include "../../coordinate/detail/tuple_algorithm.hpp"
#include <tuple>
#include <utility>


namespace ubu::detail
{


// returns (quotient, denominator, reciprocal)
template<scalar_coordinate N, scalar_coordinate D>
constexpr tuple_like auto divide_shape(const N& numerator, const D& denominator)
{
  auto n = element<0>(numerator);
  auto d = element<0>(denominator);

  return std::tuple((n+d-1)/d, d, (d+n-1)/n);
}

// returns (quotient, denominator, reciprocal)
template<coordinate N, scalar_coordinate D>
constexpr tuple_like auto divide_shape(const N& numerator, const D& denominator)
{
  using namespace std;
  using namespace ubu::detail;

  auto [quotient, d, init_and_reciprocal] = tuple_fold(tuple(tuple(), tuple(), tuple(denominator)), numerator, [](const auto& prev_result, const auto& ni)
  {
    auto [prev_quotient, prev_denominator, prev_reciprocal] = prev_result;
    auto current_denominator = tuple_last(prev_reciprocal);
    auto [current_quotient, current_denominator_, current_reciprocal] = divide_shape(ni, current_denominator);

    return tuple(tuple_append_similar_to<N>(prev_quotient, current_quotient), tuple_append_similar_to<N>(prev_denominator, current_denominator_), tuple_append_similar_to<N>(prev_reciprocal, current_reciprocal));
  });

  return tuple(quotient, d, tuple_drop_first(init_and_reciprocal));
}


} // end ubu::detail


#include "../../../detail/epilogue.hpp"

