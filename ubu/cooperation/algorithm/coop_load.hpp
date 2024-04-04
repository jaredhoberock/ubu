#pragma once

#include "../../detail/prologue.hpp"
#include "../../tensor/compose.hpp"
#include "../../tensor/coordinate/constant.hpp"
#include "../../tensor/layout/column_major.hpp"
#include "../../tensor/slice/slice.hpp"
#include "../../tensor/matrix/matrix_like.hpp"
#include "../../tensor/traits/tensor_element.hpp"
#include "../../tensor/vector/inplace_vector.hpp"
#include "../../tensor/vector/span_like.hpp"
#include "../cooperator/concepts/allocating_cooperator.hpp"
#include "../cooperator/synchronize.hpp"
#include "../uninitialized_coop_array.hpp"
#include <utility>

namespace ubu
{

// returns a copy of the first at most N * size(block) elements of source
// the copied elements are returned to each thread of block, in order
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
  using namespace ubu;

  // copy the source into a shared memory stage
  uninitialized_coop_array stage(self, source);

  // create a column-major view of the stage
  std::pair shape(constant<N>(), size(self));
  matrix_like auto stage2d = compose(stage.all(), column_major(shape));

  // get my slice of the stage
  auto my_slice = slice(stage2d, std::pair(_, id(self)));

  // copy my slice into registers
  using T = tensor_element_t<S>;
  inplace_vector<T,N> result(my_slice.begin(), my_slice.end());

  // XXX we really shouldn't have to say synchronize here
  //     what's wrong with uninitialized_coop_array's dtor synchronizing?
  //     actually the right place to put it would be coop_dealloca
  synchronize(self);

  return result;
}

} // end ubu

#include "../../detail/epilogue.hpp"

