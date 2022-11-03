#pragma once

#include "../../detail/prologue.hpp"
#include "../congruent.hpp"
#include "../coordinate.hpp"
#include "../weakly_congruent.hpp"
#include "tuple_algorithm.hpp"

#include <concepts>

namespace ubu::detail
{


// index_to_coordinate inverts the coordinate mapping of a layout 

template<scalar_coordinate I, scalar_coordinate S, scalar_coordinate D>
constexpr std::integral auto index_to_coordinate(const I& idx, const S& shape, const D& stride)
{
  return (element<0>(idx) / element<0>(stride)) % element<0>(shape);
}

template<scalar_coordinate I, nonscalar_coordinate S, nonscalar_coordinate D>
  requires (weakly_congruent<I,S> and congruent<S,D>)
constexpr congruent<S> auto index_to_coordinate(const I& idx, const S& shape, const D& stride)
{
  return detail::tuple_zip_with([&](auto& s, auto& d)
  {
    return detail::index_to_coordinate(idx,s,d);
  }, shape, stride);
}

template<nonscalar_coordinate I, nonscalar_coordinate S, nonscalar_coordinate D>
  requires (weakly_congruent<I,S> and congruent<S,D>)
constexpr congruent<S> auto index_to_coordinate(const I& idx, const S& shape, const D& stride)
{
  return detail::tuple_zip_with([](auto& i, auto& s, auto& d)
  {
    return detail::index_to_coordinate(i,s,d);
  }, idx, shape, stride);
}


} // end ubu::detail

#include "../../detail/epilogue.hpp"

