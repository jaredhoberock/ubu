#pragma once

#include "../../detail/prologue.hpp"

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
  requires (std::totally_ordered<C> and (I == 0))
constexpr bool colexicographic_coordinate_impl()
{
  return true;
}

template<std::size_t I, nonscalar_coordinate C>
  requires (std::totally_ordered<C> and (I > 0))
constexpr bool colexicographic_coordinate_impl()
{
  // create a coordinate with ones in the Ith position
  auto colexicographically_greater = zeros<C>;
  element<I>(colexicographically_greater) = ones<element_t<I,C>>;

  // create a coordinate with ones in the next leaf position to the left
  auto colexicographically_lesser = zeros<C>;
  element<I-1>(colexicographically_lesser) = ones<element_t<I-1,C>>;

  return colexicographically_lesser < colexicographically_greater and 
         colexicographically_greater > colexicographically_lesser and
         colexicographic_coordinate_impl<I-1,C>();
}


template<class T>
concept colexicographic_coordinate =
  nonscalar_coordinate<T>
  and std::totally_ordered<T>
  and colexicographic_coordinate_impl<rank_v<T> - 1,T>()
;


} // end detail
} // end ubu


#include "../../detail/epilogue.hpp"

