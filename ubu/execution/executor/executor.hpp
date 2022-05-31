#pragma once

#include "../../detail/prologue.hpp"

#include "detail/invocable_archetype.hpp"
#include "executor_of.hpp"


UBU_NAMESPACE_OPEN_BRACE


template<class E>
concept executor = executor_of<E, detail::invocable_archetype>;


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

