#pragma once

#include "../prologue.hpp"

#include "is_host.hpp"

UBU_NAMESPACE_OPEN_BRACE

namespace detail
{


constexpr bool is_device()
{
  return not is_host();
}


} // end detail

UBU_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

