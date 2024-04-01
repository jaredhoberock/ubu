#pragma once

#include "../../detail/prologue.hpp"

#include "concepts/semicooperator.hpp"
#include "id.hpp"
#include "last_id.hpp"

namespace ubu
{

template<semicooperator S>
constexpr bool is_last_in_group(S self)
{
  return id(self) == last_id(self);
}

} // end ubu

#include "../../detail/epilogue.hpp"

