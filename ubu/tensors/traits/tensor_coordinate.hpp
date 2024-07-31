#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor_like.hpp"

namespace ubu
{

template<tensor_like T>
using tensor_coordinate_t = detail::member_coordinate_or_default_t<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

