#pragma once

#include "../../detail/prologue.hpp"
#include "../views/decompose.hpp"
#include "tensor_like.hpp"

namespace ubu
{


template<class T>
concept decomposable =
  tensor_like<T> and
  requires(T tensor)
  {
    decompose(tensor);
  }
;


} // end ubu

#include "../../detail/epilogue.hpp"

