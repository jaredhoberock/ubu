#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant.hpp"
#include "../../tensors/matrices/column_major_layout.hpp"
#include "../../tensors/traits/tensor_element.hpp"
#include "../../tensors/vectors/inplace_vector.hpp"
#include "../../tensors/vectors/span_like.hpp"
#include "../../tensors/views/compose.hpp"
#include "../../tensors/views/slices/slice.hpp"
#include "../cooperators/concepts/allocating_cooperator.hpp"
#include "../cooperators/synchronize.hpp"
#include "../uninitialized_coop_array.hpp"
#include "coop_load_cyclic.hpp"
#include "coop_store_cyclic.hpp"
#include <utility>

namespace ubu
{

// returns a copy of the first at most N * size(block) elements of source
// the copied elements are returned to each thread of self, in order
//
// the source is required to be a contiguous span of memory because this
// operation is optimized for this case only.
//
// postcondition:
//   * for all threads which receive a non-empty result, result.size() == N;
//     except for the final receiving a non-empty result, result.size() <= N
//   * for all other threads, result.empty() is true
template<std::size_t N, allocating_cooperator C, span_like S>
constexpr inplace_vector<tensor_element_t<S>,N> coop_load(C self, S source)
{
  using T = tensor_element_t<S>;

  // XXX unintialized_coop_matrix would be nice
  //     its ctor would have enough knowledge to do an efficient copy
  uninitialized_coop_array<T,C,tensor_size_t<S>> stage(self, source.size());

  // load in a cyclic order from the source
  inplace_vector thread_elements = coop_load_cyclic<N>(self, source);

  // store in a cyclic order to the stage
  coop_store_cyclic(self, thread_elements, stage.all());

  synchronize(self);

  // create a 2d view of the stage
  auto stage2d = compose(stage.all(), column_major_layout(std::pair(constant<N>(), size(self))));

  // get a view of my slice of the stage
  auto my_slice = slice(stage2d, std::pair(_, id(self)));

  // load my slice
  thread_elements = load(my_slice);

  // XXX we really shouldn't have to say synchronize here
  //     what's wrong with uninitialized_coop_array's dtor synchronize?
  //     actually the right place to put it would be coop_dealloca
  synchronize(self);

  return thread_elements;
}

} // end ubu

#include "../../detail/epilogue.hpp"

