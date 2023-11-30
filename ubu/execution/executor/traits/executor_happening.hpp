#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/executor.hpp"
#include "../detail/invocable_archetype.hpp"
#include "../first_execute.hpp"

namespace ubu
{

template<executor E>
using executor_happening_t = first_execute_result_t<E, detail::invocable_archetype>;

} // end ubu

#include "../../../detail/epilogue.hpp"

