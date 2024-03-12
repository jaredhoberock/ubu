#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../cooperation/cooperator/concepts/cooperator.hpp"
#include "../../../cooperation/cooperator/traits/cooperator_thread_scope.hpp"
#include "../../../memory/buffer/empty_buffer.hpp"
#include "../../../tensor/coordinate/math/ceil_div.hpp"
#include "../../../tensor/coordinate/constant.hpp"
#include <optional>
#include <string_view>
#include <type_traits>

namespace ubu::cuda
{


constexpr auto warp_size = 32_c;
constexpr auto warp_mask = 0xFFFFFFFF_c;


struct warp_workspace
{
  empty_buffer buffer;

  struct barrier_type
  {
    constexpr static const std::string_view thread_scope = "warp";

    constexpr void arrive_and_wait() const
    {
#if defined(__CUDACC__)
      __syncwarp();
#endif
    }
  };

  barrier_type barrier;
};


// XXX we may wish to require that size(warp_like) must equal warp_size
template<class C>
concept warp_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "warp"
;


namespace detail
{


template<warp_like W, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr T shuffle_down(W, const T& x, int offset)
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
    u.words[i] = __shfl_down_sync(warp_mask, u.words[i], offset);
  }

  return u.value;
#else
  return x;
#endif
}


// this overload seems pointless, because std::optional<T> is trivially copyable
// however, including this overload actually seems to cause warp_shuffle_down to use
// fewer registers, perhaps because shuffling the optional's valid bit is cheaper than
// shuffling a full word, maybe because __shfl_down_sync can be implemented with __ballot_sync
template<warp_like W, class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr std::optional<T> shuffle_down(W self, const std::optional<T>& x, int offset)
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

  u.value = shuffle_down(self, u.value, offset);

  // communicate whether or not the value we received came from a valid object
  bool is_valid = __shfl_down_sync(warp_mask, x.has_value(), offset);

  return is_valid ? std::make_optional(u.value) : std::nullopt;
#else
  return std::nullopt;
#endif
}


} // end detail


template<warp_like W>
constexpr int synchronize_and_count(W, bool value)
{
#if defined(__CUDACC__)
  return __popc(__ballot_sync(warp_mask, value));
#else
  return -1;
#endif
}


template<warp_like W, class T, class F>
constexpr std::optional<T> coop_reduce(W self, std::optional<T> value, F binary_op)
{
  int num_values = synchronize_and_count(self, value.has_value());

  auto num_threads = warp_size;
  auto num_passes = 5_c; // this is log2(warp_size)

  if(num_values == num_threads)
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = detail::shuffle_down(self, *value, offset);
      value = binary_op(*value, other);
    }
  }
  else
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      std::optional other = detail::shuffle_down(self, value, offset);
      if((id(self) + offset < num_values) and other) *value = binary_op(*value, *other);
    }
  }

  return (id(self) == 0) ? value : std::nullopt;
}


} // end ubu::cuda

#include "../../../detail/epilogue.hpp"

