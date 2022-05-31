#pragma once

#include "../../detail/prologue.hpp"

#include "detail/invocable_archetype.hpp"
#include "executor.hpp"
#include "first_execute.hpp"

UBU_NAMESPACE_OPEN_BRACE


template<class E>
  requires executor<E>
using executor_event_t = first_execute_result_t<E, detail::invocable_archetype>;

UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

