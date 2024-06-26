#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../coordinates/concepts/congruent.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include "../../../concepts/tensor_like.hpp"
#include "../../../concepts/tensor_like_of_rank.hpp"
#include "../../../traits/tensor_element.hpp"

namespace ubu
{

template<class T>
concept layout_like =
  tensor_like<T>
  and coordinate<tensor_element_t<T>>
;

template<class L, class T>
concept layout_like_for =
  layout_like<L>
  and coordinate_for<tensor_element_t<L>, T>
;

template<class L, std::size_t R>
concept layout_like_of_rank = (layout_like<L> and tensor_rank_v<L> == R);

} // end ubu

#include "../../../../detail/epilogue.hpp"

