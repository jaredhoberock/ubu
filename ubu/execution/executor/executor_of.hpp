#pragma once

#include "../../detail/prologue.hpp"

#include "execute.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{

template<class E, class F>
concept executor_of =
  std::invocable<F> and
  std::is_nothrow_copy_constructible_v<std::remove_cvref_t<E>> and
  std::is_nothrow_destructible_v<std::remove_cvref_t<E>> and
  std::equality_comparable<E> and
  requires(E e, F f){ ubu::execute(e, f); }
;

} // end ubu

#include "../../detail/epilogue.hpp"

