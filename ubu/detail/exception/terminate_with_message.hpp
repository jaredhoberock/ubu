#pragma once

#include "../prologue.hpp"

#include "terminate.hpp"
#include <cstdio>


namespace ubu::detail
{


inline void terminate_with_message(const char* message) noexcept
{
  printf("%s\n", message);

  terminate();
}


} // end ubu::detail


#include "../epilogue.hpp"

