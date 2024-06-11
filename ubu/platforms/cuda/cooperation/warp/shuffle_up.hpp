#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/integrals/ceil_div.hpp"
#include "warp_like.hpp"
#include <concepts>
#include <optional>
#include <type_traits>

namespace ubu::cuda
{

template<warp_like W, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr T shuffle_up(W, int offset, const T& x)
{ 
#if defined(__CUDACC__)
  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union U
  {
    int words[num_words];
    T value;
    constexpr U(){} // this ctor allows the T member w/o trivial ctor
  } u;

  u.value = x;

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_up_sync(warp_mask, u.words[i], offset);
  }

  return u.value;
#else
  return x;
#endif
}


// this overload seems pointless, because std::optional<T> is trivially copyable
// however, including this overload actually seems to cause shuffle_up to use
// fewer registers, perhaps because shuffling the optional's valid bit is cheaper than
// shuffling a full word, maybe because __shfl_up_sync can be implemented with __ballot_sync
template<warp_like W, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr std::optional<T> shuffle_up(W self, int offset, const std::optional<T>& x)
{
#if defined(__CUDACC__)
  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union U
  {
    int words[num_words];
    T value;
    constexpr U(){} // this ctor allows the T member w/o trivial ctor
  } u;

  if(x)
  {
    u.value = *x;
  }

  u.value = shuffle_up(self, offset, u.value);

  // communicate whether or not the value we received came from a valid object
  bool is_valid = __shfl_up_sync(warp_mask, x.has_value(), offset);

  return is_valid ? std::make_optional(u.value) : std::nullopt;
#else
  return std::nullopt;
#endif
}


} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"


