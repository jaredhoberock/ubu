#pragma once

#include "../detail/prologue.hpp"
#include "constant_valued.hpp"

namespace ubu
{

template<class T>
concept dynamic_valued = not constant_valued<T>;

} // end ubu

#include "../detail/epilogue.hpp"

