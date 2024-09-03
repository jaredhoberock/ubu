#pragma once

#include "../../detail/prologue.hpp"
#include "../traits/tensor_element.hpp"
#include "nested_tensor_like.hpp"

namespace ubu
{

template<class T>
concept nested_tensor_like =
  tensor_like<T>
  and tensor_like<tensor_element_t<T>>
;

} // end ubu

#include "../../detail/epilogue.hpp"

