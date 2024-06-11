#pragma once

#include "../../../detail/prologue.hpp"

#include "coordinate.hpp"
#include "subdimensional.hpp"

namespace ubu
{


template<class T1, class T2>
concept superdimensional = subdimensional<T2,T1>;


} // end ubu

#include "../../../detail/epilogue.hpp"

