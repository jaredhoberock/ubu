#pragma once

#include "../detail/prologue.hpp"

#include <concepts>


ASPERA_NAMESPACE_OPEN_BRACE


struct always_complete_event
{
  constexpr void wait() const {}

  constexpr static always_complete_event make_independent_event()
  {
    return {};
  }

  always_complete_event make_contingent_event(const always_complete_event&...) const
  {
    return {};
  }
};


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

