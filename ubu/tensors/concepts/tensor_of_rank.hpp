#pragma once

#include "../../detail/prologue.hpp"
#include "../traits/tensor_rank.hpp"
#include "tensor.hpp"
#include <concepts>

namespace ubu
{

template<class T, std::size_t R>
concept tensor_of_rank = (tensor<T> and tensor_rank_v<T> == R);

} // end ubu

#include "../../detail/epilogue.hpp"

