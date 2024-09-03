#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor.hpp"
#include "../coordinates/traits/rank.hpp"
#include "tensor_shape.hpp"

namespace ubu
{

template<tensor T>
inline constexpr auto tensor_rank_v = rank_v<tensor_shape_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

