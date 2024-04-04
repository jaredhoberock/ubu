#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/constant_valued.hpp"
#include "../../miscellaneous/smaller.hpp"
#include "../../tensor/fancy_span.hpp"
#include "../../tensor/matrix/contiguous_column_major_matrix_like.hpp"
#include "../../tensor/matrix/height.hpp"
#include "../../tensor/matrix/matrix_like.hpp"
#include "../../tensor/shape/shape_element.hpp"
#include "../../tensor/slice/slice.hpp"
#include "../../tensor/traits/tensor_element.hpp"
#include "../../tensor/vector/inplace_vector.hpp"
#include "../cooperator/concepts/allocating_cooperator.hpp"
#include "../cooperator/concepts/cooperator.hpp"
#include "coop_load.hpp"
#include <utility>

namespace ubu
{

template<cooperator C, matrix_like M>
  requires constant_valued<shape_element_t<0,M>>
constexpr inplace_vector<tensor_element_t<M>, height_v<M>> coop_load_columns(C self, M source)
{
  if constexpr (allocating_cooperator<C> and contiguous_column_major_matrix_like<M>)
  {
    // in this special case, we can use the entire group to optimize loads
    // the following assumes that M is a particular type of ubu::view
    auto size = smaller(source.tensor().size(), source.layout().size());
    return coop_load<height_v<M>>(self, fancy_span(source.tensor().data(), size));
  }
  else
  {
    // just have each thread load their slice sequentially from the source
    auto my_slice = slice(source, std::pair(_, id(self)));
    return {my_slice.begin(), my_slice.end()};
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

