#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinates/element.hpp"
#include "../concepts/tensor.hpp"
#include "tensor_coordinate.hpp"
#include <utility>

namespace ubu
{

template<tensor T>
using tensor_reference_t = element_reference_t<T, tensor_coordinate_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

