#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/tuples.hpp"
#include "../../../../tensors/concepts/tensor_like.hpp"
#include "../../../../tensors/concepts/tensor_like_of.hpp"
#include "../../../../tensors/coordinates/concepts/congruent.hpp"
#include "../../../../tensors/traits/tensor_element.hpp"
#include "../../../../tensors/traits/tensor_shape.hpp"
#include "../../../causality/happening.hpp"
#include <tuple>

namespace ubu
{

template<class T>
concept asynchronous_allocation =
  tuples::pair_like<T>
  and happening<tuples::first_t<T>>
  and tensor_like<tuples::second_t<T>>
;

template<class Pair, class Element, class Shape>
concept asynchronous_tensor_like =
  asynchronous_allocation<Pair>
  and tensor_like_of<tuples::first_t<Pair>, Element>
  and congruent<tensor_shape_t<tuples::second_t<Pair>>, Shape>
;

} // end ubu

#include "../../../../detail/epilogue.hpp"

