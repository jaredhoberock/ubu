#pragma once

#include "../detail/prologue.hpp"

#include <concepts>


namespace ubu
{


struct past_event
{
  constexpr void wait() const {}

  constexpr static past_event first_cause()
  {
    return {};
  }

  past_event because_of(const past_event&...) const
  {
    return {};
  }
};


} // end ubu


#include "../detail/epilogue.hpp"

