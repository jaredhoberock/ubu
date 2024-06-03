#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant.hpp"
#include "../../tensor/layout/column_major.hpp"
#include "../../tensor/matrix/matrix_like.hpp"
#include "../../tensor/traits/tensor_element.hpp"
#include "../../tensor/vector/inplace_vector.hpp"
#include "../../tensor/vector/span_like.hpp"
#include "../../tensor/views/compose.hpp"
#include "../../tensor/views/slice/slice.hpp"
#include "../cooperator/concepts/allocating_cooperator.hpp"
#include "../cooperator/synchronize.hpp"
#include "../uninitialized_coop_array.hpp"
#include <concepts>
#include <utility>

namespace ubu
{

// copies destination.size() elements from input, in order
//
// the destination is required to be a contiguous span of memory because this
// operation is optimized for this case only
//
// precondition:
//   * for all threads contributing a non-empty input, input.size() == N;
//     except for the final thread contributing a non-empty input, input.size() <= N
//   * for all other threads, input.empty() shall be true
//   * destination.size() is exactly the sum of all input.size()
//   * the above imply that destination.size() shall be <= N * size(block)
template<allocating_cooperator C, class T, std::size_t N, span_like D>
  requires std::convertible_to<T,tensor_element_t<D>>
constexpr void coop_store(C self, const inplace_vector<T,N>& input, D destination)
{
  // cooperatively copy into a shared memory stage
  uninitialized_coop_array<T,C> stage(self, destination.size());

  // create a column-major view of the stage
  std::pair shape(constant<N>(), size(self));
  matrix_like auto matrix = compose(stage.all(), column_major(shape));

  // get my slice of the stage
  auto my_slice = slice(matrix, std::pair(_, id(self)));

  // copy my values into the stage
  input.store(my_slice);
  synchronize(self);

  coop_copy(self, stage.all(), destination);

  // XXX we really shouldn't have to say synchronize here
  //     what's wrong with uninitialized_coop_array's dtor synchronizing?
  //     actually the right place to put it would be coop_dealloca
  synchronize(self);
}

} // end ubu

#include "../../detail/epilogue.hpp"

