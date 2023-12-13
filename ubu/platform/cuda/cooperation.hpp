#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection.hpp"
#include "../../memory/buffer/empty_buffer.hpp"
#include "../../cooperation/cooperator/basic_cooperator.hpp"
#include "../../cooperation/cooperator/concepts/cooperator.hpp"
#include "../../cooperation/cooperator/cooperator_thread_scope.hpp"
#include <cmath>
#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>


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


struct block_workspace
{
  // XXX we should use small_span or similar with int size
  std::span<std::byte> buffer;

  struct barrier_type
  {
    constexpr static const std::string_view thread_scope = "block";

    constexpr void arrive_and_wait() const
    {
#if defined(__CUDACC__)
      __syncthreads();
#endif
    }
  };

  barrier_type barrier;
};


struct device_workspace
{
  constexpr static const std::string_view thread_scope = "device";

  // XXX we should use small_span or similar with int size
  std::span<std::byte> buffer;
  block_workspace local_workspace;

  constexpr device_workspace(std::span<std::byte> outer_buffer)
  {
#if defined(__CUDACC__)
    if UBU_TARGET(ubu::detail::is_device())
    {
      // count the number of dynamically-allocated shared memory bytes
      unsigned int dynamic_smem_size;
      asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_smem_size));

      // create workspace
      extern __shared__ std::byte inner_buffer[];
      buffer = outer_buffer;
      local_workspace.buffer = std::span(inner_buffer, dynamic_smem_size);
    }
#endif
  }
};


template<class C>
concept warp_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "warp"
;

template<class C>
concept block_like =
  cooperator<C>
  and cooperator_thread_scope_v<C> == "block"
;

// overload descend_with_group_coord for 1D block_like groups
// this allows us to get a warp from a block which doesn't happen
// to have a hierarchical workspace
// returns the pair (warp_cooperator, which_warp)
template<block_like B>
  requires (rank_v<shape_t<B>> == 1)
constexpr auto descend_with_group_coord(B block)
{
  constexpr int warp_size = 32;
  int lane = coord(block) % warp_size;
  int which_warp = coord(block) / warp_size;
  return std::pair(basic_cooperator(lane, warp_size, warp_workspace{}), which_warp);
}


namespace detail
{


template<std::integral T>
constexpr T ceil_div(const T& dividend, const T& divisor)
{
  return (dividend + divisor - 1) / divisor;
}


template<class T>
concept plain_old_data = std::is_trivial_v<T> and std::is_standard_layout_v<T>;


template<plain_old_data T>
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


template<plain_old_data T>
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


template<warp_like W, class T, class F>
constexpr std::optional<T> coop_reduce(W self, std::optional<T> value, int num_values, F binary_op)
{
  // XXX we need to be able to say this in a constexpr
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


namespace detail
{


template<warp_like W, class T>
constexpr int coop_count_values(W, const std::optional<T>& maybe_value)
{
  return __popc(__ballot_sync(0xFFFFFFFF, maybe_value.has_value()));
}


} // end detail


template<warp_like W, class T, class F>
constexpr std::optional<T> coop_reduce(W self, const std::optional<T>& value, F binary_op)
{
  return coop_reduce(self, value, detail::coop_count_values(self, value), binary_op);
}


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

