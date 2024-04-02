#pragma once

#include "../../detail/prologue.hpp"
#include "../shape/shape.hpp"
#include "matrix_like.hpp"

namespace ubu
{


// XXX we only really require the 0th mode of shape to be constant
template<matrix_like M>
  requires constant_shaped<M>
constexpr inline auto matrix_height_v = get<0>(shape_v<M>);

} // end ubu

#include "../../detail/epilogue.hpp"

