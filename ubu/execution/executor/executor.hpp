#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/first_cause.hpp"
#include "../../causality/happening.hpp"
#include "detail/invocable_archetype.hpp"
#include "execute_after.hpp"
#include <concepts>
#include <type_traits>


namespace ubu
{

template<class E, class F>
concept executor_of =
  std::invocable<F>
  and std::equality_comparable<E>

  and requires(E e)
  {
    {first_cause(e)} -> happening;
  }

  and requires(E e, const first_cause_result_t<E>& before, F f)
  {
    {ubu::execute_after(e, before, f)} -> happening;
  }
;

template<class E>
concept executor = executor_of<E, detail::invocable_archetype>;

} // end ubu

#include "../../detail/epilogue.hpp"

