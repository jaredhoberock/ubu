#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant_valued.hpp"
#include "../shapes/shape.hpp"
#include "../shapes/shape_element.hpp"
#include "matrix_like.hpp"

namespace ubu
{

template<matrix_like M>
using matrix_width_t = shape_element_t<1,M>;

template<matrix_like M>
  requires constant_valued<matrix_width_t<M>>
constexpr inline auto matrix_width_v = shape_element_v<1,M>;

} // end ubu

#include "../../detail/epilogue.hpp"

