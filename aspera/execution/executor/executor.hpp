#pragma once

#include "../../detail/prologue.hpp"

#include "detail/invocable_archetype.hpp"
#include "executor_of.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


template<class E>
concept executor = executor_of<E, detail::invocable_archetype>;


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

