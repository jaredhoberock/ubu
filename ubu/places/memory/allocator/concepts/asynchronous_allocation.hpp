#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensor/coordinate/detail/tuple_algorithm.hpp"
#include "../../../../tensor/shape/shape.hpp"
#include "../../../../tensor/vector/span_like.hpp"
#include "../../../causality/happening.hpp"
#include <tuple>

namespace ubu
{

template<class T>
concept asynchronous_allocation =
  detail::pair_like<T>
  and happening<std::tuple_element_t<0,T>>
  and span_like<std::tuple_element_t<1,T>>
;

template<class T, class S>
concept asynchronous_allocation_congruent_with =
  asynchronous_allocation<T>
  and coordinate<S>
  and congruent<S, shape_t<std::tuple_element_t<1,T>>>
;

} // end ubu

#include "../../../../detail/epilogue.hpp"

