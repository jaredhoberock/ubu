#pragma once

#include "../../detail/prologue.hpp"

#include "../../tensor/coordinate/concepts/integral_like.hpp"
#include "../../tensor/shape/shape.hpp"
#include "../../tensor/shape/shape_size.hpp"
#include "concepts/semicooperator.hpp"

namespace ubu
{

template<semicooperator C>
constexpr integral_like auto size(const C& self)
{
  return shape_size(shape(self));
}

} // end ubu

#include "../../detail/epilogue.hpp"

