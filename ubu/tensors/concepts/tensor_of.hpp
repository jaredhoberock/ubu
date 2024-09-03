#pragma once

#include "../../detail/prologue.hpp"
#include "../traits/tensor_element.hpp"
#include "tensor.hpp"
#include <concepts>

namespace ubu
{

template<class T, class U>
concept tensor_of =
  tensor<T>
  and std::same_as<tensor_element_t<T>,U>
;

} // end ubu

#include "../../detail/epilogue.hpp"

