#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor.hpp"
#include "../shapes/shape.hpp"

namespace ubu
{

template<tensor T>
using tensor_shape_t = shape_t<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

