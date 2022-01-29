#pragma once

#include "../prologue.hpp"

#include "terminate.hpp"
#include <cstdio>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


inline void terminate_with_message(const char* message) noexcept
{
  printf("%s\n", message);

  terminate();
}


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

