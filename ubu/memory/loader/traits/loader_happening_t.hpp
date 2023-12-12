#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/initial_happening.hpp"
#include "../loader.hpp"

namespace ubu
{

template<loader E>
using loader_happening_t = initial_happening_result_t<E>;

} // end ubu

#include "../../../detail/epilogue.hpp"

