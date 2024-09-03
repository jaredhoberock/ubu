#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/nested_tensor.hpp"
#include "inner_tensor.hpp"
#include "tensor_shape.hpp"

namespace ubu
{

template<nested_tensor T>
using inner_tensor_shape_t = tensor_shape_t<inner_tensor_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

