#pragma once

#include "../../../detail/prologue.hpp"

#include "pointer_like.hpp"

namespace ubu
{


template<pointer_like P>
using pointer_pointee_t = typename std::pointer_traits<P>::element_type;


} // end ubu

#include "../../../detail/epilogue.hpp"

