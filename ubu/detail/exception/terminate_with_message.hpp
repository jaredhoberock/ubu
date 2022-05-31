#pragma once

#include "../prologue.hpp"

#include "terminate.hpp"
#include <cstdio>


UBU_NAMESPACE_OPEN_BRACE


namespace detail
{


inline void terminate_with_message(const char* message) noexcept
{
  printf("%s\n", message);

  terminate();
}


} // end detail


UBU_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

