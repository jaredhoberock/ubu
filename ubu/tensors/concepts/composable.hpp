#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/element.hpp"
#include "../traits/tensor_element.hpp"
#include "tensor_like.hpp"

namespace ubu
{

// XXX there are other kinds of compositions we might consider
//     for example, if B doesn't have a shape, but A does,
//     we might still consider that a legal composition
//
//     IOW, B could be an invocable and A could be a tensor_like
//     the resulting composition's shape would simply be A's shape
//
// XXX I think this concept should simply check that compose(a,b) works
template<class A, class B>
concept composable =
  tensor_like<B> and 
  requires(A a, tensor_element_t<B> coord)
  {
    element(a, coord);
  }
;

} // end ubu

#include "../../detail/epilogue.hpp"

