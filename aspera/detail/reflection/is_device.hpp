#pragma once

#include "../prologue.hpp"

#include "is_host.hpp"

constexpr bool is_device()
{
  return not is_host();
}

#include "../epilogue.hpp"

