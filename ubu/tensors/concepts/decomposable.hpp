#pragma once

#include "../../detail/prologue.hpp"
#include "../views/decompose.hpp"
#include "tensor.hpp"

namespace ubu
{


template<class T>
concept decomposable =
  tensor<T> and
  requires(T t)
  {
    decompose(t);
  }
;


} // end ubu

#include "../../detail/epilogue.hpp"

