#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/constant_valued.hpp"
#include "../views/layouts/concepts/layout.hpp"
#include "../views/layouts/strides/stride_element.hpp"
#include "matrix.hpp"

namespace ubu
{

// XXX we probably also require that the layout be compact
//     but that's not easily noticed at compile time
template<class T>
concept column_major_like =
  layout<T>
  and matrix<T>
  and constant_valued<stride_element_t<0,T>>
  and stride_element_v<0,T> == 1
;

} // end ubu

#include "../../detail/epilogue.hpp"

