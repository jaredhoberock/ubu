#pragma once

#include "../../detail/prologue.hpp"
#include "../traits/tensor_element.hpp"
#include "tensor.hpp"

namespace ubu
{

template<class T>
concept nested_tensor =
  tensor<T>
  and tensor<tensor_element_t<T>>
;

} // end ubu

#include "../../detail/epilogue.hpp"

