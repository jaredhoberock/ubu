#pragma once

#include "../../detail/prologue.hpp"
#include "sized_tensor.hpp"
#include "view.hpp"

namespace ubu
{

template<class T>
concept sized_view = view<T> and sized_tensor<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

