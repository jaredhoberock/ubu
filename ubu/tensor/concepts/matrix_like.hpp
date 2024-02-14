#pragma once

#include "../../detail/prologue.hpp"
#include "tensor_like_of_rank.hpp"
#include <concepts>

namespace ubu
{

template<class T>
concept matrix_like = ubu::tensor_like_of_rank<T,2>;

} // end ubu

#include "../../detail/epilogue.hpp"

