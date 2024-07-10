#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/integral_like.hpp"
#include "../../utilities/integrals/size.hpp"
#include "../../tensors/shapes/shape.hpp"
#include "../../tensors/shapes/shape_size.hpp"
#include "../concepts/semicooperator.hpp"

namespace ubu
{

template<semicooperator C>
constexpr integral_like auto tag_invoke(decltype(ubu::size), const C& self)
{
  return shape_size(shape(self));
}

} // end ubu

#include "../../detail/epilogue.hpp"

