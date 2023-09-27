#pragma once

#include "../../detail/prologue.hpp"

#include "../grid.hpp"

namespace ubu
{

template<class T>
concept layout =
  grid<T>
  and coordinate<grid_element_t<T>>
;

} // end ubu

#include "../../detail/epilogue.hpp"

