#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor_like.hpp"
#include "../coordinates/traits/rank.hpp"
#include "tensor_shape.hpp"

namespace ubu
{

template<tensor_like T>
inline constexpr auto tensor_rank_v = rank_v<tensor_shape_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

