#pragma once

#include "../../detail/prologue.hpp"

#include "concepts/semicooperator.hpp"
#include "id.hpp"

namespace ubu
{

template<ubu::semicooperator S>
constexpr bool is_leader(S self)
{
  return id(self) == 0;
}

} // end ubu

#include "../../detail/epilogue.hpp"

