#pragma once

#include "../detail/prologue.hpp"

#include <concepts>


namespace ubu
{


struct past_event
{
  constexpr void wait() const {}

  constexpr bool has_happened() const
  {
    return true;
  }

  constexpr static past_event initial_happening()
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

