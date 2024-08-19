#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "../traits/rank.hpp"
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
  if constexpr (rank_v<T1> == 0 and rank_v<T2> == 0)
  {
    // terminal case 0: two rank-0 types are weakly congruent
    return true;
  }
  else if constexpr (rank_v<T1> == 1 and rank_v<T2> == 1)
  {
    // terminal case 1: two rank-1 types are weakly congruent
    return true;
  }
  else if constexpr (rank_v<T1> == 1 and rank_v<T2> > 1)
  {
    // recursive case 0: a rank-1 type T1 is weakly congruent to a rank-n type T2
    // if T1 is weakly congruent to all of T2's elements
    auto all_weakly_congruent = []<std::size_t...I>(std::index_sequence<I...>)
    {
      return (... and is_weakly_congruent<T1,tuples::element_t<I,T2>>());
    };

    return all_weakly_congruent(tuples::indices_v<T2>);
  }
  else if constexpr (rank_v<T1> == rank_v<T2>)
  {
    // recursive case 1: a rank-n type T1 is weakly congruent to a rank-n type T2
    // if T1[i] is weakly congruent to T2[i]
    auto all_weakly_congruent = []<std::size_t...I>(std::index_sequence<I...>)
    {
      return (... and is_weakly_congruent<tuples::element_t<I,T1>,tuples::element_t<I,T2>>());
    };

    return all_weakly_congruent(tuples::indices_v<T1>);
  }
  else
  {
    // terminal case 2: a rank-m type is not weakly congruent to a rank-n type
    return false;
  }
}


} // end detail


template<class T1, class T2>
concept weakly_congruent =
  semicoordinate<T1>
  and semicoordinate<T2>
  and detail::is_weakly_congruent<T1,T2>();
;


} // end ubu

#include "../../../detail/epilogue.hpp"

