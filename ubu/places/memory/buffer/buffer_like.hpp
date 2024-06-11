#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../tensor/traits/tensor_element.hpp"
#include "../../../tensor/vectors/span_like.hpp"
#include <cstddef>

namespace ubu
{


template<class T>
concept buffer_like = span_like<T> and std::same_as<tensor_element_t<T>, std::byte>;


} // end ubu

#include "../../../detail/epilogue.hpp"

