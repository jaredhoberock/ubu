#pragma once

#include "../../../detail/prologue.hpp"
#include "../concepts/cooperator.hpp"
#include "../size.hpp"

namespace ubu
{

template<cooperator C>
using cooperator_size_t = decltype(size(std::declval<C>()));

} // end ubu

#include "../../../detail/epilogue.hpp"

