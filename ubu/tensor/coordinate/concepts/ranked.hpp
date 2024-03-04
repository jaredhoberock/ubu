#pragma once

#include "../../../detail/prologue.hpp"

#include "../detail/tuple_algorithm.hpp"
#include "../traits/rank.hpp"

namespace ubu
{
namespace detail
{


template<class T>
constexpr bool is_ranked()
{
  if constexpr(not static_rank<T>)
  {
    // if T doesn't have a static rank, then T is not ranked
    return false;
  }
  else if constexpr(tuple_like<T>)
  {
    // all tuple_like have a static rank

    // T is ranked if all of its tuple elements are also ranked
    auto all_elements_are_ranked = []<std::size_t... I>(std::index_sequence<I...>)
    {
      return (... and is_ranked<std::tuple_element_t<I,std::remove_cvref_t<T>>>());
    };

    return all_elements_are_ranked(tuple_indices<T>);
  }
  else
  {
    // T has a static rank and it is not a tuple, so T is ranked
    return true;
  }
}


} // end detail


// XXX if "ranked" is not a good name, then "semicoordinate" may be an alternative
template<class T>
concept ranked = detail::is_ranked<T>();


} // end ubu


#include "../../../detail/epilogue.hpp"

