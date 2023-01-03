#pragma once

#include "../../../detail/prologue.hpp"

#include "../coordinate.hpp"
#include "../element.hpp"
#include "../ones.hpp"
#include "../rank.hpp"
#include "../zeros.hpp"
#include <concepts>


namespace ubu
{
namespace detail
{


template<std::size_t I, nonscalar_coordinate C>
  requires (std::totally_ordered<C> and (I == rank_v<C> - 1))
constexpr bool lexicographic_coordinate_impl()
{
  return true;
}

template<std::size_t I, nonscalar_coordinate C>
  requires (std::totally_ordered<C> and (I < rank_v<C> - 1))
constexpr bool lexicographic_coordinate_impl()
{
  // create a coordinate with ones in the ith position
  auto lexicographically_greater = zeros<C>;
  element<I>(lexicographically_greater) = ones<element_t<I,C>>;

  // create a coordinate with ones in the next position to the right
  auto lexicographically_lesser = zeros<C>;
  element<I+1>(lexicographically_lesser) = ones<element_t<I+1,C>>;

  return lexicographically_lesser < lexicographically_greater and 
         lexicographically_greater > lexicographically_lesser and
         lexicographic_coordinate_impl<I+1,C>();
}


template<class T>
concept lexicographic_coordinate =
  nonscalar_coordinate<T>
  and std::totally_ordered<T>
  and lexicographic_coordinate_impl<0,T>()
;


} // end detail
} // end ubu

#include "../../../detail/epilogue.hpp"

