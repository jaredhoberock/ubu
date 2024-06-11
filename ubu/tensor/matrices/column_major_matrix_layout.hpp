#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant_valued.hpp"
#include "../views/layout/layout.hpp"
#include "../views/layout/stride/stride_element.hpp"
#include "matrix_like.hpp"

namespace ubu
{

// XXX we probably also require that the layout be compact
//     but that's not easily noticed at compile time
template<class T>
concept column_major_matrix_layout =
  layout<T>
  and matrix_like<T>
  and constant_strided<T>
  and stride_element_v<0,T> == 1
;

} // end ubu

#include "../../detail/epilogue.hpp"

