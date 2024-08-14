#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "../traits/rank.hpp"
#include "coordinate.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


template<coordinate L, coordinate R>
constexpr bool is_subdimensional()
{
  if constexpr (unary_coordinate<L>)
  {
    // terminal case 0: a unary_coordinate is subdimensional to anything that is not a nullary_coordinate
    return not nullary_coordinate<R>;
  }
  else if constexpr (rank_v<R> < rank_v<L>)
  {
    // terminal case 1: R has lower rank than L
    return false;
  }
  else
  {
    // recursive case: rank_v<L> <= rank_v<R>
    auto all_elements_are_subdimensional = []<std::size_t... I>(std::index_sequence<I...>)
    {
      return (... and is_subdimensional<tuples::element_t<I,L>, tuples::element_t<I,R>>());
    };

    return all_elements_are_subdimensional(tuples::indices_v<L>);
  }
}


} // end detail


// subdimensional is a recursive concept so it is implemented with a constexpr function
template<class T1, class T2>
concept subdimensional =
  coordinates<T1,T2>
  and detail::is_subdimensional<T1,T2>()
;


} // end ubu

#include "../../../detail/epilogue.hpp"

