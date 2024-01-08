#pragma once

#include "../../detail/prologue.hpp"
#include "../view.hpp"
#include <span>

namespace ubu
{

// XXX crop_bottom should be a CPO
// it isn't at the moment because the default implementation has not been written yet

template<class T>
constexpr T* crop_bottom(T* ptr, std::ptrdiff_t new_origin)
{
  return ptr + new_origin;
}

template<class T>
constexpr std::span<T> crop_bottom(const std::span<T>& s, std::size_t new_origin)
{
  // don't return an out-of-range subspan
  return new_origin <= s.size() ? s.subspan(new_origin) : std::span<T>();
}

template<grid G>
constexpr void crop_bottom(const G& g, const grid_shape_t<G>& new_origin)
{
  static_assert(sizeof(G) != 0, "crop_bottom(grid): Not yet implemented.");
}

} // end ubu

#include "../../detail/epilogue.hpp"

