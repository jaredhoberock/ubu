#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../cooperation/cooperator/concepts/cooperator.hpp"
#include "../../../cooperation/cooperator/traits/cooperator_thread_scope.hpp"
#include "../../../memory/buffer/empty_buffer.hpp"
#include "../../../tensor/coordinate/math/ceil_div.hpp"
#include <optional>
#include <string_view>
#include <type_traits>

namespace ubu::cuda
{


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


template<class C>
concept warp_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "warp"
;


namespace detail
{


template<class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr T shuffle_down(const T& x, int offset, int width)
{ 
#if defined(__CUDACC__)
  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;
  u.value = x;

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_down_sync(__activemask(), u.words[i], offset, width);
  }

  return u.value;
#else
  return {};
#endif
}


template<class T>
  requires std::is_trivially_copy_constructible_v<T>
constexpr std::optional<T> shuffle_down(const std::optional<T>& x, int offset, int width)
{
#if defined(__CUDACC__)
  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;

  if(x)
  {
    u.value = *x;
  }

  for(int i= 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_down_sync(__activemask(), u.words[i], offset, width);
  }

  // communicate whether or not the words we shuffled came from a valid object
  bool is_valid = x ? true : false;
  is_valid = __shfl_down_sync(__activemask(), is_valid, offset, width);

  return is_valid ? std::make_optional(u.value) : std::nullopt;
#else
  return {};
#endif
}


} // end detail


template<warp_like W>
constexpr int synchronize_and_count(W, bool value)
{
#if defined(__CUDACC__)
  return __popc(__ballot_sync(0xFFFFFFFF, value));
#else
  return -1;
#endif
}


template<warp_like W, class T, class F>
constexpr std::optional<T> coop_reduce(W self, std::optional<T> value, F binary_op)
{
  int num_values = synchronize_and_count(self, value.has_value());

  // XXX warp_like should maybe require that size(self) is constexpr and equal to 32
  // constexpr int num_threads = size(self);
  constexpr int warp_size = 32;
  constexpr int num_threads = warp_size;
  constexpr int num_passes = 5; // this is log2(warp_size)

  if(num_values == num_threads)
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = detail::shuffle_down(*value, offset, num_threads);
      value = binary_op(*value, other);
    }
  }
  else
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      std::optional other = detail::shuffle_down(value, offset, num_threads);
      if((id(self) + offset < num_values) and other) *value = binary_op(*value, *other);
    }
  }

  return (id(self) == 0) ? value : std::nullopt;
}


} // end ubu::cuda

#include "../../../detail/epilogue.hpp"

