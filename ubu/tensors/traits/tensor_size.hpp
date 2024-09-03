#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/integrals/size.hpp"
#include "../concepts/sized_tensor.hpp"
#include <ranges>

namespace ubu
{

template<sized_tensor T>
using tensor_size_t = size_result_t<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

