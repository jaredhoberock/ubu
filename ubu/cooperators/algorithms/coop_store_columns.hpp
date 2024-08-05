#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/constant_valued.hpp"
#include "../../tensors/matrices/contiguous_column_major_matrix_like.hpp"
#include "../../tensors/matrices/matrix_height.hpp"
#include "../../tensors/matrices/matrix_like.hpp"
#include "../../tensors/shapes/shape_element.hpp"
#include "../../tensors/traits/tensor_element.hpp"
#include "../../tensors/vectors/inplace_vector.hpp"
#include "../../tensors/views/decompose.hpp"
#include "../../tensors/views/slices/slice.hpp"
#include "../concepts/allocating_cooperator.hpp"
#include "../concepts/cooperator.hpp"
#include "coop_store.hpp"
#include <utility>

namespace ubu
{

template<cooperator C, class T, std::size_t N, matrix_like M>
  requires constant_valued<matrix_height_t<M>> and (matrix_height_v<M> == N) and std::convertible_to<T,tensor_element_t<M>>
constexpr void coop_store_columns(C self, const inplace_vector<T,N>& this_column, M destination)
{
  if constexpr (allocating_cooperator<C> and contiguous_column_major_matrix_like<M>)
  {
    // in this special case, we can use the entire group to accelerate stores
    auto [span, _] = decompose(destination);
    coop_store(self, this_column, span);
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

