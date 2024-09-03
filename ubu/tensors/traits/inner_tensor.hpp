#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/nested_tensor.hpp"
#include "inner_tensor.hpp"
#include "tensor_element.hpp"

namespace ubu
{

template<nested_tensor T>
using inner_tensor_t = tensor_element_t<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

