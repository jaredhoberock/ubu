#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integral/integral_like.hpp"
#include "../../tensor/coordinates/coord.hpp"
#include "../../tensor/shapes/shape.hpp"
#include "../../tensor/views/layouts/column_major.hpp"
#include "concepts/semicooperator.hpp"
#include <concepts>

namespace ubu
{

template<semicooperator C>
constexpr integral_like auto id(const C& self)
{
  column_major layout(shape(self));
  return layout[coord(self)];
}

} // end ubu

#include "../../detail/epilogue.hpp"

