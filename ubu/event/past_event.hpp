#pragma once

#include "../detail/prologue.hpp"

#include <concepts>


namespace ubu
{


struct past_event
{
  constexpr void wait() const {}

  constexpr static past_event make_independent_event()
  {
    return {};
  }

  past_event make_dependent_event(const past_event&...) const
  {
    return {};
  }
};


} // end ubu


#include "../detail/epilogue.hpp"

