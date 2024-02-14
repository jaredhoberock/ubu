#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/tensor_like.hpp"
#include "tensor_reference.hpp"
#include <type_traits>

namespace ubu
{

template<tensor_like T>
using tensor_element_t = std::remove_cvref_t<tensor_reference_t<T>>;

} // end ubu

#include "../../detail/epilogue.hpp"

