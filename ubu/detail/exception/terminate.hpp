#pragma once

#include "../prologue.hpp"

#include "../reflection.hpp"

#include <exception>

UBU_NAMESPACE_OPEN_BRACE


namespace detail
{


inline void terminate() noexcept
{
  if UBU_TARGET(is_device())
  {
    asm("trap;");
  }
  else
  {
    std::terminate();
  }
}


} // end detail

UBU_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

