#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensors/concepts/tensor_like.hpp"
#include "../../../../tensors/concepts/tensor_like_of.hpp"
#include "../../../../tensors/coordinates/concepts/congruent.hpp"
#include "../../../../tensors/coordinates/detail/tuple_algorithm.hpp"
#include "../../../../tensors/traits/tensor_element.hpp"
#include "../../../../tensors/traits/tensor_shape.hpp"
#include "../../../causality/happening.hpp"
#include <tuple>

namespace ubu
{

template<class T>
concept asynchronous_allocation =
  detail::pair_like<T>
  and happening<std::tuple_element_t<0,T>>
  and tensor_like<std::tuple_element_t<1,T>>
;

template<class Pair, class Element, class Shape>
concept asynchronous_tensor_like =
  asynchronous_allocation<Pair>
  and tensor_like_of<std::tuple_element_t<1,Pair>, Element>
  and congruent<tensor_shape_t<std::tuple_element_t<1,Pair>>, Shape>
;

} // end ubu

#include "../../../../detail/epilogue.hpp"

