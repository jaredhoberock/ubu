#pragma once

#include "../../../detail/prologue.hpp"

#include "../detail/tuple_algorithm.hpp"
#include "equal_rank.hpp"
#include "semicoordinate.hpp"
#include <tuple>
#include <type_traits>
#include <utility>


namespace ubu
{
namespace detail
{


template<semicoordinate T1, semicoordinate T2>
constexpr bool is_weakly_congruent()
{
  if constexpr(rank_v<T1> == 1)
  {
    // terminal case 1: T1 has rank 1
    // rank-1 types are weakly congruent to all other ranked types
    return true;
  }
  else if constexpr(not equal_rank<T1,T2>)
  {
    // terminal case 2: T1 has rank > 1 and it differs from T2's rank
    return false;
  }
  else
  {
    // recursive case: T1 and T2 have equal rank
    // recurse across all elements
    auto all_weakly_congruent = []<std::size_t...I>(std::index_sequence<I...>)
    {
      using U1 = std::remove_cvref_t<T1>;
      using U2 = std::remove_cvref_t<T2>;

      return (... and is_weakly_congruent<std::tuple_element_t<I,U1>, std::tuple_element_t<I,U2>>());
    };

    return all_weakly_congruent(tuple_indices<T1>);
  }
}


} // end detail


template<class T1, class T2>
concept weakly_congruent =
  semicoordinate<T1>
  and semicoordinate<T2>
  and detail::is_weakly_congruent<T1,T2>()
;


} // end ubu

#include "../../../detail/epilogue.hpp"

