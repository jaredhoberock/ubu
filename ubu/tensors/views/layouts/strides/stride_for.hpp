#pragma once

#include "../../../../detail/prologue.hpp"

#include "apply_stride.hpp"
#include <utility>


namespace ubu
{


template<class S, class C>
concept stride_for = requires(S stride, C coord)
{
  { apply_stride(std::forward<S>(stride), std::forward<C>(coord)) } -> coordinate;
};


} // end ubu

#include "../../../../detail/epilogue.hpp"

