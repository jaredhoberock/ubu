#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/concepts/congruent.hpp"
#include "../../coordinate/concepts/coordinate.hpp"
#include "../../coordinate/concepts/equal_rank.hpp"
#include "../../coordinate/detail/as_integral_like.hpp"
#include "../../coordinate/detail/tuple_algorithm.hpp"
#include <concepts>


namespace ubu::detail
{


template<scalar_coordinate A, scalar_coordinate B>
constexpr integral_like auto compose_strides(const A& a, const B& b)
{
  return as_integral_like(a) * as_integral_like(b);
}


template<nonscalar_coordinate A, nonscalar_coordinate B>
  requires equal_rank<A,B>
constexpr congruent<B> auto compose_strides(const A& a, const B& b);


template<coordinate A, nonscalar_coordinate B>
  requires (rank_v<A> < rank_v<B>)
constexpr congruent<B> auto compose_strides(const A& a, const B& b)
{
  return detail::tuple_zip_with(b, [&](const auto& bi)
  {
    return compose_strides(a, bi);
  });
}


template<nonscalar_coordinate A, nonscalar_coordinate B>
  requires equal_rank<A,B>
constexpr congruent<B> auto compose_strides(const A& a, const B& b)
{
  return detail::tuple_zip_with(a, b, [](const auto& ai, const auto& bi)
  {
    return compose_strides(ai, bi);
  });
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

