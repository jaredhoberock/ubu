#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/constant_valued.hpp"
#include "../shapes/shape.hpp"
#include "../shapes/shape_element.hpp"
#include "matrix.hpp"

namespace ubu
{

template<matrix M>
using matrix_height_t = shape_element_t<0,M>;

template<matrix M>
  requires constant_valued<matrix_height_t<M>>
constexpr inline auto matrix_height_v = shape_element_v<0,M>;

} // end ubu

#include "../../detail/epilogue.hpp"

