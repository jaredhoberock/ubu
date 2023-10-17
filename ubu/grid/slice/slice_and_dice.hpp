#pragma once

#include "../../detail/prologue.hpp"
#include "../grid.hpp"
#include "dice.hpp"
#include "slice.hpp"
#include "slicer.hpp"

namespace ubu
{

// XXX in principle, slice_and_dice could be a CPO

// returns the pair (slice(g,katana), dice(g,katana))
template<grid G, slicer_for<grid_shape_t<G>> S>
constexpr auto slice_and_dice(const G& g, const S& katana)
{
  return std::make_pair(slice(g,katana), dice(g,katana));
}

} // end ubu

#include "../../detail/epilogue.hpp"
