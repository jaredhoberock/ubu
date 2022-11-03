#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "element.hpp"
#include "same_rank.hpp"
#include <concepts>
#include <type_traits>
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


// terminal case 2: both arguments are rank 1 coordinates
template<coordinate_of_rank<1> T1, coordinate_of_rank<1> T2>
constexpr bool is_weakly_congruent()
{
  return true;
}


// terminal case 3: the first argument is rank 1 coordinate and the second is any tuple_like_coordinate
template<coordinate_of_rank<1> T1, tuple_like_coordinate T2>
constexpr bool is_weakly_congruent()
{
  return true;
}


// recursive case: both arguments are tuple_like_coordinate and their ranks are the same
// this is a forward declaration for is_weakly_congruent_recursive_impl
template<tuple_like_coordinate T1, tuple_like_coordinate T2>
  requires (rank_v<T1> != 1 and same_rank<T1,T2>)
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

// recursive case: two non-congruent coordinates
template<tuple_like_coordinate T1, tuple_like_coordinate T2>
  requires (rank_v<T1> != 1 and same_rank<T1,T2>)
constexpr bool is_weakly_congruent()
{
  return is_weakly_congruent_recursive_impl<T1,T2>(std::make_index_sequence<rank_v<T1>>{});
}


} // end detail


// we use remove_cvref_t because std::integral doesn't like references
template<class T1, class T2>
concept weakly_congruent = detail::is_weakly_congruent<std::remove_cvref_t<T1>,std::remove_cvref_t<T2>>();


} // end ubu

#include "../detail/epilogue.hpp"

