#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integrals/integral_like.hpp"
#include "../../tensors/coordinates/coord.hpp"
#include "../../tensors/shapes/shape.hpp"
#include "../../tensors/views/layouts/column_major.hpp"
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

