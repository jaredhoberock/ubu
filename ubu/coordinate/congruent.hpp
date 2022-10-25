#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "detail/number.hpp"
#include "rank.hpp"
#include "same_rank.hpp"
#include <concepts>
#include <type_traits>


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


// terminal case 2: both arguments are the same kind of number
template<number T1, number T2>
  requires same_kind_of_number<T1,T2>
constexpr bool are_congruent()
{
  return true;
}


// forward declaration of recursive case
template<coordinate T1, coordinate T2>
  requires (not number<T1> and
            not number<T2> and
            same_rank<T1,T2>)
constexpr bool are_congruent();


template<coordinate T1, coordinate T2>
constexpr bool are_congruent_recursive_impl(std::index_sequence<>)
{
  return true;
}


template<coordinate T1, coordinate T2, std::size_t Index, std::size_t... Indices>
  requires (not number<T1> and
            not number<T2> and
            same_rank<T1,T2>)
constexpr bool are_congruent_recursive_impl(std::index_sequence<Index, Indices...>)
{
  // check the congruency of the first element of each coordinate and recurse to the rest of the elements
  return are_congruent<element_t<Index,T1>, element_t<Index,T2>>() and are_congruent_recursive_impl<T1,T2>(std::index_sequence<Indices...>{});
}


// recursive case: neither arguments are numbers but both are coordinates
//                 and their ranks are the same
template<coordinate T1, coordinate T2>
  requires (not number<T1> and
            not number<T2> and
            same_rank<T1,T2>)
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


// we use remove_cvref_t because std::integral and std::floating_point don't like references
template<class T1, class T2, class... Types>
concept congruent = detail::are_congruent<std::remove_cvref_t<T1>,std::remove_cvref_t<T2>,std::remove_cvref_t<Types>...>();


} // end ubu

#include "../detail/epilogue.hpp"

