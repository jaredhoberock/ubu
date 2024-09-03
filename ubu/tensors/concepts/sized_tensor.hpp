#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/integrals/size.hpp"
#include "tensor.hpp"
#include <ranges>

namespace ubu
{

template<class T>
concept sized_tensor = tensor<T> and sized<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

