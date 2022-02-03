#pragma once

#include "../prologue.hpp"

ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
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


} // end detail

ASPERA_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

