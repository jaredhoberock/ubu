#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/integral/ceil_div.hpp"
#include "warp_like.hpp"
#include <concepts>
#include <optional>
#include <type_traits>

namespace ubu::cuda
{


template<warp_like W, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr T broadcast(W self, int broadcaster, const T& message)
{
#if defined(__CUDACC__)
  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union U
  {
    int words[num_words];
    T value;
    constexpr U(){} // this ctor allows the T member w/o trivial ctor
  } u;

  u.value = message;

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_sync(warp_mask, u.words[i], broadcaster);
  }

  return u.value;
#else
  return message;
#endif
}

template<warp_like W, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr std::optional<T> broadcast(W self, int broadcaster, const std::optional<T>& message)
{
#if defined(__CUDACC__)
  // if the broadcaster has no message, exit early
  if(not __shfl_sync(warp_mask, message.has_value(), broadcaster))
  {
    return std::nullopt;
  }

  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union U
  {
    int words[num_words];
    T value;
    constexpr U(){} // this ctor allows the T member w/o trivial ctor
  } u;

  if(message)
  {
    u.value = *message;
  }

  // broadcast the message
  return broadcast(self, broadcaster, u.value);
#else
  return std::nullopt;
#endif
}

} // end ubu::cuda

#include "../../../../detail/epilogue.hpp"

