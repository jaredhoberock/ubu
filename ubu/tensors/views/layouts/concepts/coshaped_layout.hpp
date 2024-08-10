#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../coordinates/concepts/congruent.hpp"
#include "../../../traits/tensor_element.hpp"
#include "../coshape.hpp"
#include "layout.hpp"

namespace ubu
{

template<class T>
concept coshaped_layout =
  layout<T>
  and coshaped<T>
  and congruent<coshape_t<T>, tensor_element_t<T>>
;

} // end ubu

#include "../../../../detail/epilogue.hpp"
