#pragma once

#include "../../detail/prologue.hpp"
#include "tensor_like.hpp"
#include <ranges>
#include <type_traits>

namespace ubu
{

// XXX consider naming this tensor_view
// XXX consider relaxing is_trivially_copy_constructible_v
// we reuse std::ranges::enable_view rather than define our own so that types don't need to specialize multiple enable_view templates
template<class T>
concept view = tensor_like<T> and std::is_trivially_copy_constructible_v<T> and std::ranges::enable_view<T>;

} // end ubu

#include "../../detail/epilogue.hpp"

