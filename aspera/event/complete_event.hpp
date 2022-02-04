#pragma once

#include "../detail/prologue.hpp"

#include <concepts>


ASPERA_NAMESPACE_OPEN_BRACE


struct complete_event
{
  constexpr void wait() {}

  constexpr static complete_event make_complete_event()
  {
    return {};
  }

  // for any given executor, and a set of complete_events, we can always
  // create a complete_event contingent on all of them
  // this is a friend function to avoid ambiguity with the CPO aspera::contingent_on
  template<class E>
  constexpr friend complete_event contingent_on(E&&, const complete_event&...)
  {
    return {};
  }
};


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

