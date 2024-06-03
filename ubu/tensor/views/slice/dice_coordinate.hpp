#pragma once

#include "../../../detail/prologue.hpp"

#include "opposite_slicer.hpp"
#include "slice_coordinate.hpp"
#include "slicer.hpp"

namespace ubu
{

template<slicer C, slicer_for<C> K>
constexpr slicer auto dice_coordinate(const C& coord, const K& katana)
{
  return slice_coordinate(coord, opposite_slicer(coord, katana));
}

} // end ubu

#include "../../../detail/epilogue.hpp"

