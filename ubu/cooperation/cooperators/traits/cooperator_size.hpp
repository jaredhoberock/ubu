#pragma once

#include "../../../detail/prologue.hpp"
#include "../concepts/semicooperator.hpp"
#include "../size.hpp"

namespace ubu
{

template<semicooperator C>
using cooperator_size_t = decltype(size(std::declval<C>()));

} // end ubu

#include "../../../detail/epilogue.hpp"

