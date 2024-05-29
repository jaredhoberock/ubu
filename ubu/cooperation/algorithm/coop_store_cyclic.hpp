#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/constant.hpp"
#include "../../tensor/compose.hpp"
#include "../../tensor/layout/row_major.hpp"
#include "../../tensor/slice/slice.hpp"
#include "../../tensor/traits/tensor_element.hpp"
#include "../../tensor/vector/inplace_vector.hpp"
#include "../../tensor/vector/vector_like.hpp"
#include <concepts>
#include <utility>

namespace ubu
{

// copies destination.size() elements from input, cyclically
//
// that is, each thread of self contributes elements from input to destination coordinates
//
//   id(self), id(self) + 1 * size(self), ..., id(self) + (N-1) * size(self)
//
template<class T, std::size_t N, semicooperator C, vector_like D>
  requires std::convertible_to<T,tensor_element_t<D>>
constexpr void coop_store_cyclic(C self, const inplace_vector<T,N>& input, D destination)
{
  // create a matrix view of the destination
  // each thread's stores will begin at id(self) and stride through destination with a stride of size(self)
  auto matrix = compose(destination, row_major(std::pair(constant<N>(), size(self))));

  // get this thread's slice of the view
  auto thread_slice = slice(matrix, std::pair(_, id(self)));

  // store from the input
  input.store(thread_slice);
}

} // end ubu

#include "../../detail/epilogue.hpp"

