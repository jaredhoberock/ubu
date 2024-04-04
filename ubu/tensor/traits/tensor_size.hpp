#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/sized_tensor_like.hpp"
#include <ranges>

namespace ubu
{

template<sized_tensor_like T>
using tensor_size_t = decltype(std::ranges::size(std::declval<T>()));

} // end ubu

#include "../../detail/epilogue.hpp"

