#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor.hpp"

namespace ubu
{

template<tensor T>
using tensor_coordinate_t = detail::member_coordinate_or_default_t<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

