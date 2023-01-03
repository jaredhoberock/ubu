#pragma once

#include "../../detail/prologue.hpp"

#include "coordinate.hpp"
#include "element.hpp"
#include "same_rank.hpp"
#include <concepts>
#include <utility>


namespace ubu
{
namespace detail
{


// terminal case 1: both arguments are unrelated types 
template<class T1, class T2>
constexpr bool is_weakly_congruent()
{
  return false;
}


// terminal case 2: both arguments are scalar
template<scalar_coordinate T1, scalar_coordinate T2>
constexpr bool is_weakly_congruent()
{
  return true;
}


// terminal case 3: the first argument is scalar and the second is nonscalar
template<scalar_coordinate T1, nonscalar_coordinate T2>
constexpr bool is_weakly_congruent()
{
  return true;
}


// recursive case: both arguments are nonscalar and their ranks are the same
// this is a forward declaration for is_weakly_congruent_recursive_impl
template<nonscalar_coordinate T1, nonscalar_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool is_weakly_congruent();


template<coordinate T1, coordinate T2>
constexpr bool is_weakly_congruent_recursive_impl(std::index_sequence<>)
{
  return true;
}


template<coordinate T1, coordinate T2, std::size_t I, std::size_t... Is>
constexpr bool is_weakly_congruent_recursive_impl(std::index_sequence<I,Is...>)
{
  // check the weak congruency of the first element of each coordinate and recurse to the rest of the elements
  return is_weakly_congruent<element_t<I,T1>, element_t<I,T2>>() and is_weakly_congruent_recursive_impl<T1,T2>(std::index_sequence<Is...>{});
}

// recursive case: two nonscalar coordinates
template<nonscalar_coordinate T1, nonscalar_coordinate T2>
  requires same_rank<T1,T2>
constexpr bool is_weakly_congruent()
{
  return is_weakly_congruent_recursive_impl<T1,T2>(std::make_index_sequence<rank_v<T1>>{});
}


} // end detail


// weakly_congruent is recursive concept so it is implemented with SFINAE
template<class T1, class T2>
concept weakly_congruent = detail::is_weakly_congruent<T1,T2>();


} // end ubu

#include "../../detail/epilogue.hpp"

