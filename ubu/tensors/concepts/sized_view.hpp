#pragma once

#include "../../detail/prologue.hpp"
#include "sized_tensor_like.hpp"
#include "view.hpp"

namespace ubu
{

template<class T>
concept sized_view = view<T> and sized_tensor_like<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

