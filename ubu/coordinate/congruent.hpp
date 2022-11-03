#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "rank.hpp"
#include "same_rank.hpp"
#include "weakly_congruent.hpp"
#include <concepts>
#include <utility>


namespace ubu
{


namespace detail
{


// terminal case 1: both arguments are unrelated types
template<class T1, class T2>
constexpr bool are_congruent()
{
  return false;
}


// terminal case 2: both arguments are rank 1 coordinates
template<coordinate_of_rank<1> T1, coordinate_of_rank<1> T2>
constexpr bool are_congruent()
{
  return true;
}


// forward declaration of recursive case
template<tuple_like_coordinate T1, tuple_like_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool are_congruent();


template<tuple_like_coordinate T1, tuple_like_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool are_congruent_recursive_impl(std::index_sequence<>)
{
  return true;
}


template<tuple_like_coordinate T1, tuple_like_coordinate T2, std::size_t Index, std::size_t... Indices>
  requires same_rank<T1,T2>
constexpr bool are_congruent_recursive_impl(std::index_sequence<Index, Indices...>)
{
  // check the congruency of the first element of each coordinate and recurse to the rest of the elements
  return are_congruent<element_t<Index,T1>, element_t<Index,T2>>() and are_congruent_recursive_impl<T1,T2>(std::index_sequence<Indices...>{});
}


// recursive case: neither arguments are rank 1 but both are coordinates
//                 and their ranks are the same
template<tuple_like_coordinate T1, tuple_like_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool are_congruent()
{
  return are_congruent_recursive_impl<T1,T2>(std::make_index_sequence<rank_v<T1>>{});
}


// variadic case
// requiring a third argument disambiguates this function from the others above
template<coordinate T1, coordinate T2, coordinate T3, coordinate... Types>
constexpr bool are_congruent()
{
  return are_congruent<T1,T2>() and are_congruent<T1,T3,Types...>();
}


} // end detail


template<class T1, class T2, class... Types>
concept congruent =
  weakly_congruent<T1,T2>
  and (... and weakly_congruent<T1,Types>)
  and detail::are_congruent<T1,T2,Types...>()
;


} // end ubu

#include "../detail/epilogue.hpp"

