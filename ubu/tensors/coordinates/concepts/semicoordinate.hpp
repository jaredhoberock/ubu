#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../miscellaneous/tuples.hpp"
#include "../traits/rank.hpp"

namespace ubu
{
namespace detail
{


template<class T>
constexpr bool is_semicoordinate()
{
  if constexpr(not static_rank<T>)
  {
    // if T doesn't have a static rank, then T is not a semicoordinate
    return false;
  }
  else if constexpr(tuples::tuple_like<T>)
  {
    // all tuple_like have a static rank

    // T is a semicoordinate if all of its tuple elements are semicoordinates
    auto all_elements_are_semicoordinates = []<std::size_t... I>(std::index_sequence<I...>)
    {
      return (... and is_semicoordinate<std::tuple_element_t<I,std::remove_cvref_t<T>>>());
    };

    return all_elements_are_semicoordinates(tuples::indices_v<T>);
  }
  else
  {
    // T has a static rank and it is not a tuple, so T is ranked
    return true;
  }
}


} // end detail


template<class T>
concept semicoordinate = detail::is_semicoordinate<T>();


} // end ubu


#include "../../../detail/epilogue.hpp"

