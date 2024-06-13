#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/integrals/size.hpp"
#include "../concepts/sized_tensor_like.hpp"
#include <ranges>

namespace ubu
{

template<sized_tensor_like T>
using tensor_size_t = size_result_t<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

