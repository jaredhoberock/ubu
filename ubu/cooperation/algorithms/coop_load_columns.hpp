#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/constant_valued.hpp"
#include "../../tensors/matrices/contiguous_column_major_matrix_like.hpp"
#include "../../tensors/matrices/matrix_height.hpp"
#include "../../tensors/matrices/matrix_like.hpp"
#include "../../tensors/shapes/shape_element.hpp"
#include "../../tensors/traits/tensor_element.hpp"
#include "../../tensors/vectors/inplace_vector.hpp"
#include "../../tensors/views/slices/slice.hpp"
#include "../cooperators/concepts/allocating_cooperator.hpp"
#include "../cooperators/concepts/cooperator.hpp"
#include "coop_load.hpp"
#include <utility>

namespace ubu
{

template<cooperator C, matrix_like M>
  requires constant_valued<matrix_height_t<M>>
constexpr inplace_vector<tensor_element_t<M>, matrix_height_v<M>> coop_load_columns(C self, M source)
{
  if constexpr (allocating_cooperator<C> and contiguous_column_major_matrix_like<M>)
  {
    // in this special case, we can use the entire group to optimize loads
    // the following assumes that M is a particular type of composed_view
    // s.t. source.tensor() is span_like
    return coop_load<matrix_height_v<M>>(self, source.span());
  }
  else
  {
    // just have each thread load their slice sequentially from the source
    auto my_slice = slice(source, std::pair(_, id(self)));
    return load(my_slice);
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

