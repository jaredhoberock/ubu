#pragma once

#include "../../detail/prologue.hpp"

#include "execute.hpp"
#include <concepts>
#include <type_traits>


UBU_NAMESPACE_OPEN_BRACE

template<class E, class F>
concept executor_of =
  std::invocable<F> and
  std::is_nothrow_copy_constructible_v<std::remove_cvref_t<E>> and
  std::is_nothrow_destructible_v<std::remove_cvref_t<E>> and
  std::equality_comparable<E> and
  requires(E e, F f){ UBU_NAMESPACE::execute(e, f); }
;

UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

