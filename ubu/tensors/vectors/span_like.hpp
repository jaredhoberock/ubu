#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/view.hpp"
#include "contiguous_vector_like.hpp"
#include <concepts>

namespace ubu
{

template<class T>
concept span_like = contiguous_vector_like<T> and view<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

