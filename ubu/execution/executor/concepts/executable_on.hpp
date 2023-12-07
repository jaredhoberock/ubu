#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include "../../../causality/initial_happening.hpp"
#include "../execute_after.hpp"
#include <concepts>

namespace ubu
{


// XXX consider a reorganization that would organize the happening and the invocable
//     into a single "executable" concept
template<class F, class E, class H = initial_happening_result_t<E>>
concept executable_on =
  ubu::happening<H>
  and requires(E ex, H before, F f)
  {
    { execute_after(ex, before, f) } -> happening;
  }
;


} // end ubu

#include "../../../detail/epilogue.hpp"

