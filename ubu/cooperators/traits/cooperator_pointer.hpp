#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/allocating_cooperator.hpp"
#include "../primitives/coop_alloca.hpp"
#include <utility>

namespace ubu
{

template<allocating_cooperator C>
using cooperator_pointer_t = decltype(coop_alloca(std::declval<C>(), std::declval<int>()));

} // end ubu

#include "../../detail/epilogue.hpp"

