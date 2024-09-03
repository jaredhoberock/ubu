#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor_of_rank.hpp"
#include <concepts>

namespace ubu
{

template<class T>
concept vector_like = ubu::tensor_of_rank<T,1>;

} // end ubu

#include "../../detail/epilogue.hpp"

