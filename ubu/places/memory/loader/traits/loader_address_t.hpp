#pragma once

#include "../../../../detail/prologue.hpp"

#include "../loader.hpp"

namespace ubu
{

template<loader L>
using loader_address_t = typename std::remove_cvref_t<L>::address_type;

} // end ubu

#include "../../../../detail/epilogue.hpp"

