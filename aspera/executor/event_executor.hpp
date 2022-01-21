#pragma once

#include "../detail/prologue.hpp"

#include <concepts>
#include "../event/event.hpp"
#include "executor.hpp"


ASPERA_NAMESPACE_OPEN_BRACE

template<class E>
concept event_executor =
  executor<E> and
  requires(E e)
  {
    { ASPERA_NAMESPACE::execute(e, detail::invocable_archetype{}) } -> event;
  }
;

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

