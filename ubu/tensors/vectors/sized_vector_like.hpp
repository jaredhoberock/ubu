#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/sized_tensor.hpp"
#include "vector_like.hpp"
#include <ranges>

namespace ubu
{

template<class V>
concept sized_vector_like = vector_like<V> and sized_tensor<V>;

} // end ubu

#include "../../detail/epilogue.hpp"

