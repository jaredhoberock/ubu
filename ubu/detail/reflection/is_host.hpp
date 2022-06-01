#pragma once

#include "../prologue.hpp"


namespace ubu::detail
{


constexpr bool is_host()
{
  bool result = true;

#if defined(__circle_lang__)
  if target(not __is_host_target)
  {
    result = false;
  }
#endif

  return result;
}


} // end ubu::detail


#include "../epilogue.hpp"

