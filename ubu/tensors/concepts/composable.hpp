#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/element.hpp"
#include "../traits/tensor_element.hpp"
#include "../element_exists.hpp"
#include "tensor.hpp"

namespace ubu
{

// XXX there are other kinds of compositions we might consider
//     for example, if B doesn't have a shape, but A does,
//     we might still consider that a legal composition
//
//     IOW, B could be an invocable and A could be a tensor
//     the resulting composition's shape would simply be A's shape
template<class A, class B>
concept composable =
  tensor<B> and 
  requires(A a, tensor_element_t<B> coord)
  {
    element(a, coord);
    element_exists(a, coord);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

