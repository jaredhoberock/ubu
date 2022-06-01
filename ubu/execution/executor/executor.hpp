#pragma once

#include "../../detail/prologue.hpp"

#include "detail/invocable_archetype.hpp"
#include "executor_of.hpp"

namespace ubu
{


template<class E>
concept executor = executor_of<E, detail::invocable_archetype>;


} // end ubu

#include "../../detail/epilogue.hpp"

