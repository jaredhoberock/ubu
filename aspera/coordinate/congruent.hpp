#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "detail/number.hpp"
#include "size.hpp"
#include <concepts>
#include <type_traits>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


// terminal case 1: both arguments are integers
template<std::integral T1, std::integral T2>
constexpr bool are_congruent()
{
  return true;
}


// terminal case 2: both arguments are floating point
template<std::floating_point T1, std::floating_point T2>
constexpr bool are_congruent()
{
  return true;
}


// terminal case 3: the first argument is integral and the second is floating point
template<std::integral T1, std::floating_point T2>
constexpr bool are_congruent()
{
  return false;
}


// terminal case 4: the first argument is floating point and the second is integral
template<std::floating_point T1, std::integral T2>
constexpr bool are_congruent()
{
  return false;
}


// terminal case 5: the first argument is a number and the second is not
template<number T1, not_a_number T2>
constexpr bool are_congruent()
{
  return false;
}


// terminal case 6: the first argument is not a number and the second is a number
template<not_a_number T1, number T2>
constexpr bool are_congruent()
{
  return false;
}


// terminal case 7: neither arguments are coordinates
template<class T1, class T2>
  requires (!coordinate<T1> and !coordinate<T2>)
constexpr bool are_congruent()
{
  return false;
}


// terminal case 8: neither arguments are numbers but both are coordinates
//                  but their sizes differ
template<coordinate T1, coordinate T2>
  requires (not_a_number<T1> and
            not_a_number<T2> and
            size_v<T1> != size_v<T2>)
constexpr bool are_congruent()
{
  return false;
}


// forward declaration of recursive case
template<coordinate T1, coordinate T2>
  requires (not_a_number<T1> and
            not_a_number<T2> and
            size_v<T1> == size_v<T2>)
constexpr bool are_congruent();


template<coordinate T1, coordinate T2>
constexpr bool are_congruent_recursive_impl(std::index_sequence<>)
{
  return true;
}


template<coordinate T1, coordinate T2, std::size_t Index, std::size_t... Indices>
  requires (not_a_number<T1> and
            not_a_number<T2> and
            size_v<T1> == size_v<T2>)
constexpr bool are_congruent_recursive_impl(std::index_sequence<Index, Indices...>)
{
  // check the congruency of the first element of each coordinate and recurse to the rest of the elements
  return are_congruent<element_t<Index,T1>, element_t<Index,T2>>() and are_congruent_recursive_impl<T1,T2>(std::index_sequence<Indices...>{});
}


// recursive case: neither arguments are numbers but both are coordinates
//                 and their sizes are the same
template<coordinate T1, coordinate T2>
  requires (not_a_number<T1> and
            not_a_number<T2> and
            size_v<T1> == size_v<T2>)
constexpr bool are_congruent()
{
  return are_congruent_recursive_impl<T1,T2>(std::make_index_sequence<size_v<T1>>{});
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


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

