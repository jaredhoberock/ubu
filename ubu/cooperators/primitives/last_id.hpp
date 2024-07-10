#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/constant.hpp"
#include "../concepts/semicooperator.hpp"
#include "size.hpp"

namespace ubu
{

template<semicooperator C>
constexpr integral_like auto last_id(const C& self)
{
  return size(self) - 1_c;
}

} // end ubu

#include "../../detail/epilogue.hpp"

