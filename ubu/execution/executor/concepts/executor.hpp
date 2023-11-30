#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/first_cause.hpp"
#include "../../../causality/happening.hpp"
#include "../detail/invocable_archetype.hpp"
#include "executable_on.hpp"
#include <concepts>


namespace ubu
{

template<class E, class H, class F>
concept dependent_executor_of =
  std::equality_comparable<E>
  and executable_on<F, E, H>
;

template<class E, class F>
concept executor_of =
  requires(E e)
  {
    first_cause(e);
  }
  and dependent_executor_of<E, first_cause_result_t<E>, F>
;

template<class E>
concept executor = executor_of<E, detail::invocable_archetype>;

} // end ubu

#include "../../../detail/epilogue.hpp"

