#pragma once

#include "../../../detail/prologue.hpp"

#include "../rank.hpp"
#include "coordinate.hpp"
#include "same_rank.hpp"
#include "weakly_congruent.hpp"
#include <concepts>
#include <tuple>
#include <type_traits>
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


// terminal case 2: both arguments are scalar coordinates
template<scalar_coordinate T1, scalar_coordinate T2>
constexpr bool are_congruent()
{
  return true;
}


// forward declaration of recursive case
template<nonscalar_coordinate T1, nonscalar_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool are_congruent();


template<nonscalar_coordinate T1, nonscalar_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool are_congruent_recursive_impl(std::index_sequence<>)
{
  return true;
}


template<nonscalar_coordinate T1, nonscalar_coordinate T2, std::size_t Index, std::size_t... Indices>
  requires same_rank<T1,T2>
constexpr bool are_congruent_recursive_impl(std::index_sequence<Index, Indices...>)
{
  // check the congruency of the first element of each coordinate and recurse to the rest of the elements
  return are_congruent<std::tuple_element_t<Index,T1>, std::tuple_element_t<Index,T2>>() and are_congruent_recursive_impl<T1,T2>(std::index_sequence<Indices...>{});
}


// recursive case: both arguments are nonscalar and their ranks are the same
template<nonscalar_coordinate T1, nonscalar_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool are_congruent()
{
  return are_congruent_recursive_impl<std::remove_cvref_t<T1>,std::remove_cvref_t<T2>>(std::make_index_sequence<rank_v<T1>>{});
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

#include "../../../detail/epilogue.hpp"

