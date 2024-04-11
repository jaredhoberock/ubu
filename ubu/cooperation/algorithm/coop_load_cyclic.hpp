#pragma once

#include "../../detail/prologue.hpp"
#include "../../tensor/compose.hpp"
#include "../../tensor/layout/strided_layout.hpp"
#include "../../tensor/vector/inplace_vector.hpp"
#include "../../tensor/vector/vector_like.hpp"
#include "../cooperator/concepts/semicooperator.hpp"
#include <utility>

namespace ubu
{

// returns a copy of the first at most N * size(block) elements of source
// the copied elements are returned to each thread of self in cyclic order
//
// that is, each thread of self receives elements from source with coordinates
//
//   id(self), id(self) + 1 * size(self), ..., id(self) + (N-1) * size(self)
//
template<std::size_t N, semicooperator C, vector_like S>
constexpr inplace_vector<tensor_element_t<S>,N> coop_load_cyclic(C self, S source)
{
  // create a 2d, cyclic view of the source
  // each thread's loads will begin at id(self) and stride through source with a stride of size(self)
  auto cycled = compose(source, strided_layout(std::pair(N, size(self)), std::pair(size(self), 1_c)));

  // get this thread's slice of the view
  auto thread_slice = slice(cycled, std::pair(_, id(self)));

  // load the slice into an inplace_vector
  return {from_vector_like, thread_slice};
}

} // end ubu

#include "../../detail/epilogue.hpp"

