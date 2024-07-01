#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/tuples.hpp"
#include "../traits/rank.hpp"
#include "equal_rank.hpp"
#include "semicoordinate.hpp"
#include "weakly_congruent.hpp"
#include <tuple>
#include <type_traits>
#include <utility>


namespace ubu
{
namespace detail
{


template<semicoordinate T1, semicoordinate T2>
constexpr bool are_congruent()
{
  if constexpr(equal_rank<T1,T2>)
  {
    if constexpr(rank_v<T1> == 1)
    {
      // terminal case 1: both types have rank 1
      return true;
    }
    else
    {
      // recursive case
      auto elements_are_congruent = []<std::size_t...I>(std::index_sequence<I...>)
      {
        using U1 = std::remove_cvref_t<T1>;
        using U2 = std::remove_cvref_t<T2>;

        return (... and are_congruent<std::tuple_element_t<I,U1>, std::tuple_element_t<I,U2>>());
      };

      return elements_are_congruent(tuples::indices_v<T1>);
    }
  }
  else
  {
    // terminal case 2: the types' ranks differ
    return false;
  }
}


// variadic case
// requiring a third argument disambiguates this function from the one above
template<semicoordinate T1, semicoordinate T2, semicoordinate T3, semicoordinate... Types>
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

