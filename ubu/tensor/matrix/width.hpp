#pragma once

#include "../../detail/prologue.hpp"
#include "../shape/shape.hpp"
#include "matrix_like.hpp"

namespace ubu
{


// XXX we only really require the 1st mode of shape to be constant
template<matrix_like M>
  requires constant_shaped<M>
constexpr inline auto width_v = get<1>(shape_v<M>);

} // end ubu

#include "../../detail/epilogue.hpp"

