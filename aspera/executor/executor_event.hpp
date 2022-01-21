#pragma once

#include "../detail/prologue.hpp"

#include "detail/invocable_archetype.hpp"
#include "upstream_executor.hpp"

ASPERA_NAMESPACE_OPEN_BRACE


template<class E>
  requires upstream_executor<E>
using executor_event_t = execute_result_t<E, detail::invocable_archetype>;

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

