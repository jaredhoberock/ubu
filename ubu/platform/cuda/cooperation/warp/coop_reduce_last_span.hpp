#pragma once

#include "../../../../detail/prologue.hpp"
#include "coop_reduce.hpp"
#include "warp_like.hpp"
#include <concepts>
#include <optional>
#include <type_traits>

namespace ubu::cuda
{

template<warp_like W, class T, std::invocable<T,T> F>
  requires std::convertible_to<T, std::invoke_result_t<F,T,T>>
constexpr std::optional<T> coop_reduce_last_span(W warp, std::optional<T> value, bool span_begin, F op)
{
  // if my index is before the last span, invalidate my value
  // XXX if the position is high enough, we could use a subwarp for this
  auto pos = coop_find_last(warp, span_begin);
  return coop_reduce(warp, id(warp) < pos ? std::nullopt : value, op);
}

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

