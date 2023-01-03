#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinate/congruent.hpp"
#include "../../coordinate/coordinate.hpp"
#include "../../coordinate/detail/tuple_algorithm.hpp"
#include "../../coordinate/element.hpp"
#include "../../coordinate/same_rank.hpp"
#include <concepts>


namespace ubu::detail
{


template<scalar_coordinate A, scalar_coordinate B>
constexpr std::integral auto compose_strides(const A& a, const B& b)
{
  return element<0>(a) * element<0>(b);
}


template<nonscalar_coordinate A, nonscalar_coordinate B>
  requires same_rank<A,B>
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
  requires same_rank<A,B>
constexpr congruent<B> auto compose_strides(const A& a, const B& b)
{
  return detail::tuple_zip_with(a, b, [](const auto& ai, const auto& bi)
  {
    return compose_strides(ai, bi);
  });
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

