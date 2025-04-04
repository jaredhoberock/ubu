#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/constant.hpp"
#include "../../tensors/matrices/row_major_layout.hpp"
#include "../../tensors/vectors/inplace_vector.hpp"
#include "../../tensors/vectors/vector_like.hpp"
#include "../../tensors/views/compose.hpp"
#include "../concepts/semicooperator.hpp"
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
  // create a matrix view of the destination
  // each thread's stores will begin at id(self) and stride through source with a stride of size(self)
  auto matrix = compose(source, row_major_layout(std::pair(constant<N>(), size(self))));

  // get this thread's slice of the view
  auto thread_slice = slice(matrix, std::pair(_, id(self)));

  // load the slice into an inplace_vector
  return {from_vector_like, thread_slice};
}

} // end ubu

#include "../../detail/epilogue.hpp"

