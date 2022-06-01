#pragma once

#include "../prologue.hpp"

#include "is_host.hpp"


namespace ubu::detail
{


constexpr bool is_device()
{
  return not is_host();
}


} // end ubu::detail


#include "../epilogue.hpp"

