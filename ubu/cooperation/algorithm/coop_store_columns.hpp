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

template<cooperator C, class T, std::size_t N, matrix_like M>
  requires (height_v<M> == N) and std::convertible_to<T,tensor_element_t<M>>
constexpr void coop_store_columns(C self, const inplace_vector<T,N>& this_column, M destination)
{
  if constexpr (allocating_cooperator<C> and contiguous_column_major_matrix_like<M>)
  {
    // in this special case, we can use the entire group to accelerate stores
    // the following assumes that M is a particular type of ubu::view
    auto size = smaller(destination.tensor().size(), destination.layout().size());
    coop_store(self, this_column, fancy_span(destination.tensor().data(), size));
  }
  else
  {
    // just have each thread store their slice sequentially into the destination
    auto my_slice = slice(destination, std::pair(_, id(self)));
    this_column.store(my_slice);
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

