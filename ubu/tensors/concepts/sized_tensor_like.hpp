#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/integral/size.hpp"
#include "tensor_like.hpp"
#include <ranges>

namespace ubu
{

template<class T>
concept sized_tensor_like = tensor_like<T> and sized<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

