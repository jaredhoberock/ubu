#pragma once

#include "../../detail/prologue.hpp"

#include <concepts>
#include "execute.hpp"


ASPERA_NAMESPACE_OPEN_BRACE

template<class E, class F>
concept executor_of =
  std::invocable<F> and
  std::is_nothrow_copy_constructible_v<E> and
  std::is_nothrow_destructible_v<E> and
  std::equality_comparable<E> and
  requires(E e, F f){ ASPERA_NAMESPACE::execute(e, f); }
;

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

