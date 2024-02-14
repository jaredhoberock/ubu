#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor_like.hpp"
#include "../shape/shape.hpp"

namespace ubu
{

template<tensor_like T>
using tensor_shape_t = shape_t<T>;

template<tensor_like T>
using tensor_coordinate_t = detail::coordinate_or_default_t<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

