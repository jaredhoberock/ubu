#pragma once

#include "../prologue.hpp"

#include "is_device.hpp"


namespace ubu::detail
{


constexpr bool has_exceptions()
{
  bool result = true;

#if defined(__cpp_exceptions)
  result = __cpp_exceptions;
#endif

#if defined(__circle_lang__)
  if target(is_device())
  {
    result = false;
  }
#endif

  return result;
}


} // end ubu::detail


#include "../epilogue.hpp"

