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

template<std::integral I, std::integral S, std::integral D>
constexpr std::integral auto index_to_coordinate(const I& idx, const S& shape, const D& stride)
{
  return (idx / stride) % shape;
}

template<std::integral I, ubu::tuple_like_coordinate S, ubu::tuple_like_coordinate D>
  requires (ubu::weakly_congruent<I,S> and ubu::congruent<S,D>)
constexpr ubu::congruent<S> auto index_to_coordinate(const I& idx, const S& shape, const D& stride)
{
  return detail::tuple_zip_with([&](auto& s, auto& d)
  {
    return detail::index_to_coordinate(idx,s,d);
  }, shape, stride);
}

template<ubu::tuple_like_coordinate I, ubu::tuple_like_coordinate S, ubu::tuple_like_coordinate D>
  requires (ubu::weakly_congruent<I,S> and ubu::congruent<S,D>)
constexpr ubu::congruent<S> auto index_to_coordinate(const I& idx, const S& shape, const D& stride)
{
  return detail::tuple_zip_with([](auto& i, auto& s, auto& d)
  {
    return detail::index_to_coordinate(i,s,d);
  }, idx, shape, stride);
}



} // end ubu::detail

#include "../../detail/epilogue.hpp"

