#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../utilities/tuples.hpp"
#include "slicer.hpp"
#include "underscore.hpp"

namespace ubu
{

template<slicer C, slicer_for<C> K>
constexpr slicer auto opposite_slicer(const C& coord, const K& katana)
{
  if constexpr (detail::is_underscore_v<K>)
  {
    // terminal case 1: katana is literally the underscore

    // return coord if it is not an underscore, otherwise 0
    if constexpr (not detail::is_underscore_v<C>)
    {
      return coord;
    }
    else
    {
      return 0;
    }
  }
  else if constexpr (coordinate<K>)
  {
    // terminal case 2: katana is a coordinate, return an underscore
    return _;
  }
  else if constexpr (tuples::unit_like<K>)
  {
    // terminal case 3: katana is a unit, return a unit
    return katana;
  }
  else
  {
    // recursive case: both katana and coord are same-size tuples
    static_assert(tuples::same_size<C,K>);

    return tuples::zip_with(katana, coord, [](const auto& k, const auto& c)
    {
      return opposite_slicer(c, k);
    });
  }
}

} // end ubu

#include "../../../detail/epilogue.hpp"

