#pragma once

#include "../../../detail/prologue.hpp"

#include "../../coordinates/traits/rank.hpp"
#include "slicer.hpp"
#include <tuple>
#include <type_traits>

namespace ubu
{
namespace detail
{

template<slicer K, slicer C>
constexpr bool is_unslicer_for()
{
  // when comparing katana against coord, we consider the underscores contained in katana

  if constexpr (is_underscore_v<K>)
  {
    // when katana is _, then coord can be anything
    return true;
  }
  else if constexpr (rank_v<K> == 0)
  {
    // when katana is (), then coord must be ()
    return rank_v<C> == 0;
  }
  else
  {
    // consider the number of underscores in katana

    if constexpr (underscore_count_v<K> == 1 and rank_v<C> == 0)
    {
      // if katana contains a single underscore, then coord can be ()
      return true;
    }
    else
    {
      // otherwise, coord's rank must be equal to the number of underscores in katana
      return underscore_count_v<K> == rank_v<C>;
    }
  }
}


} // end detail


template<class K, class C>
concept unslicer_for =
  ubu::slicer<K>
  and ubu::slicer<C>
  and detail::is_unslicer_for<K,C>()
;

} // end ubu

#include "../../../detail/epilogue.hpp"

