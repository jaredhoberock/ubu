#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/ceil_div.hpp"
#include "../../utilities/integrals/size.hpp"
#include "../../tensors/traits/tensor_reference.hpp"
#include "../../tensors/concepts/tensor.hpp"
#include "../../tensors/matrices/row_major_layout.hpp"
#include "../../tensors/vectors/vector_like.hpp"
#include "../../tensors/views/compose.hpp"
#include "../../tensors/views/nestle.hpp"
#include "../concepts/cooperator.hpp"
#include "../primitives/size.hpp"
#include "../primitives/synchronize.hpp"
#include <concepts>
#include <utility>

namespace ubu
{
namespace detail
{


// XXX reorganize this into ubu/algorithms/sequential/for_each.hpp if we decide that's necessary
template<ubu::tensor T, std::invocable<ubu::tensor_reference_t<T>> F>
constexpr void for_each(T tensor, F f)
{
  for(auto&& e : tensor)
  {
    // XXX do we need to forward e?
    f(e);
  }
}


} // end detail

template<cooperator C, tensor T, std::invocable<tensor_reference_t<T>> F>
constexpr void coop_for_each(C self, T tensor, F f)
{
  if constexpr (vector_like<T>)
  {
    // fold the vector into a matrix and recurse to the else branch
    std::pair shape(ceil_div(size(tensor), size(self)), size(self));

    coop_for_each(self, compose(tensor, row_major_layout(shape)), f);
  }
  else
  {
    vector_like auto my_slice = nestle(tensor)[id(self)];

    // do my portion of the problem
    detail::for_each(my_slice, f);

    // wait for the rest of the group
    synchronize(self);
  }
}

} // end ubu

#include "../../detail/epilogue.hpp"

