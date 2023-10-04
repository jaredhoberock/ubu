#pragma once

#include "../../../detail/prologue.hpp"

#include "../bind_executable.hpp"


namespace ubu
{


template<class E, class F, class... Args>
concept bindable_for =
  requires(E executor, F f, Args... args)
  {
    bind_executable(executor, f, args...);
  }
;


} // end ubu


#include "../../../detail/epilogue.hpp"

