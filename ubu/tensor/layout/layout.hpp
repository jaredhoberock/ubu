#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinate/concepts/congruent.hpp"
#include "../coordinate/concepts/coordinate.hpp"
#include "../concepts/tensor_like.hpp"
#include "../traits/tensor_element.hpp"

namespace ubu
{

template<class T>
concept layout =
  tensor_like<T>
  and coordinate<tensor_element_t<T>>
;

template<class L, class T>
concept layout_for =
  layout<L>
  and coordinate_for<tensor_element_t<L>, T>
;

} // end ubu

#include "../../detail/epilogue.hpp"

