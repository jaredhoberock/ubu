#pragma once

#include "../../../../detail/prologue.hpp"

#include "coop_ballot.hpp"
#include "warp_like.hpp"
#include <bit>

namespace ubu::cuda
{

template<warp_like W>
constexpr std::uint32_t synchronize_and_count(W warp, bool value)
{
  return std::popcount(coop_ballot(warp, value));
}

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

