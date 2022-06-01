#pragma once

#include "../prologue.hpp"

#include "../reflection.hpp"

#include <exception>


namespace ubu::detail
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


} // end ubu::detail


#include "../epilogue.hpp"

