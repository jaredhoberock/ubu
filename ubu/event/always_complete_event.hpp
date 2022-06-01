#pragma once

#include "../detail/prologue.hpp"

#include <concepts>


namespace ubu
{


struct always_complete_event
{
  constexpr void wait() const {}

  constexpr static always_complete_event make_independent_event()
  {
    return {};
  }

  always_complete_event make_dependent_event(const always_complete_event&...) const
  {
    return {};
  }
};


} // end ubu


#include "../detail/epilogue.hpp"

