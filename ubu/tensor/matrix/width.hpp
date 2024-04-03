#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant_valued.hpp"
#include "../shape/shape.hpp"
#include "../shape/shape_element.hpp"
#include "matrix_like.hpp"

namespace ubu
{


template<matrix_like M>
  requires constant_valued<shape_element_t<1,M>>
constexpr inline auto width_v = shape_element_v<1,M>;

} // end ubu

#include "../../detail/epilogue.hpp"

