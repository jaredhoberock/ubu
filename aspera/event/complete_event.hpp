#pragma once

#include "../detail/prologue.hpp"


ASPERA_NAMESPACE_OPEN_BRACE


struct complete_event
{
  constexpr void wait() {}

  constexpr static complete_event make_complete_event()
  {
    return {};
  }
};


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

