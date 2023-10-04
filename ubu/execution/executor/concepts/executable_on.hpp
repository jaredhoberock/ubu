#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/first_cause.hpp"
#include "../../../causality/happening.hpp"
#include "../execute_after.hpp"
#include <concepts>

namespace ubu
{


template<class F, class E, class H = first_cause_result_t<E>>
concept executable_on =
  ubu::happening<H>
  and requires(E ex, H before, F f)
  {
    { execute_after(ex, before, f) } -> happening;
  }
;


} // end ubu

#include "../../../detail/epilogue.hpp"

