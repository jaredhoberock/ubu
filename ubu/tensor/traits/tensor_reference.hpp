#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinate/element.hpp"
#include "../concepts/tensor_like.hpp"
#include "tensor_coordinate.hpp"
#include <utility>

namespace ubu
{

template<tensor_like T>
using tensor_reference_t = decltype(element(std::declval<T>(), std::declval<tensor_coordinate_t<T>>()));

} // end ubu

#include "../../detail/epilogue.hpp"

