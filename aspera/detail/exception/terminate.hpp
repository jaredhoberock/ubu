#pragma once

#include "../prologue.hpp"

#include "../reflection.hpp"

#include <exception>

ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


inline void terminate() noexcept
{
  if ASPERA_TARGET(is_device())
  {
    asm("trap;");
  }
  else
  {
    std::terminate();
  }
}


} // end detail

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

