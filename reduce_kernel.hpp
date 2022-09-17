#pragma once

#include "stride.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <iterator>
#include <optional>
//#include <sm_30_intrinsics.hpp>


// we have to define our own warp intrinsics because circle can't deal with the CUDA header files where they are defined
inline __device__ unsigned my_activemask()
{
  unsigned ret;
  asm volatile ("activemask.b32 %0;" : "=r"(ret));
  return ret;
}


constexpr int warp_size = 32;


inline __device__ int my_shfl_down_sync(unsigned mask, int var, unsigned int delta, int width)
{
  extern __device__ __device_builtin__ unsigned __nvvm_shfl_down_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
  int ret;
  int c = ((warp_size-width) << 8) | 0x1f;
  ret = __nvvm_shfl_down_sync(mask, var, delta, c);
  return ret;
}


template<std::integral I>
constexpr I ceil_div(I numerator, I denominator)
{
  return (numerator + denominator - I{1}) / denominator;
}


template<class T>
concept plain_old_data = std::is_trivial_v<T> and std::is_standard_layout_v<T>;


template<std::integral I>
constexpr bool is_pow2(I x)
{
  return 0 == (x & (x - 1));
}


template<plain_old_data T>
T shuffle_down(const T& x, int offset, int width)
{ 
  constexpr std::size_t num_words = ceil_div(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;
  u.value = x;

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = my_shfl_down_sync(my_activemask(), u.words[i], offset, width);
  }

  return u.value;
}


template<plain_old_data T>
std::optional<T> shuffle_down(const std::optional<T>& x, int offset, int width)
{
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
    u.words[i] = my_shfl_down_sync(my_activemask(), u.words[i], offset, width);
  }

  // communicate whether or not the words we shuffled came from a valid object
  bool is_valid = x ? true : false;
  is_valid = my_shfl_down_sync(my_activemask(), is_valid, offset, width);

  return is_valid ? std::make_optional(u.value) : std::nullopt;
}


template<class I, class O, class F>
concept reducible =
  std::random_access_iterator<I> and
  std::regular_invocable<F, std::iter_value_t<I>, std::iter_value_t<I>> and
  std::random_access_iterator<O> and
  std::indirectly_writable<O, std::invoke_result_t<F, std::iter_value_t<I>, std::iter_value_t<I>>>
;


template<class I, class O, class F>
concept plain_old_reducible =
  reducible<I, O, F> and
  plain_old_data<std::iter_value_t<I>>
;


constexpr bool is_valid_subwarp_size(int x)
{
  return x <= warp_size and is_pow2(x);
}


// assume num_values <= num_threads
// the result is returned only to lane 0
// the result is undefined for other lanes
template<int num_threads, plain_old_data T, std::invocable<T,T> F>
  requires (is_valid_subwarp_size(num_threads) and plain_old_reducible<T*,T*,F>)
T subwarp_reduce(int lane_idx, T value, int num_values, F binary_op)
{
  constexpr int num_passes = std::log2(num_threads);

  if(num_values == num_threads)
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = shuffle_down(value, offset, num_threads);
      value = binary_op(value, other);
    }
  }
  else
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = shuffle_down(value, offset, num_threads);
      if(lane_idx + offset < num_values)
      {
        value = binary_op(value, other);
      }
    }
  }

  return value;
}


// assume num_values <= num_threads
// the result is returned only to lane 0 if num_values > 0
// std::nullopt is returned to all other lanes
template<int num_threads, plain_old_data T, std::invocable<T,T> F>
  requires (is_valid_subwarp_size(num_threads) and plain_old_reducible<T*,T*,F>)
std::optional<T> subwarp_reduce(int lane_idx, std::optional<T> value, int num_values, F binary_op)
{
  constexpr int num_passes = std::log2(num_threads);

  if(num_values == num_threads)
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      T other = shuffle_down(*value, offset, num_threads);
      value = binary_op(*value, other);
    }
  }
  else
  {
    for(int pass = 0; pass != num_passes; ++pass)
    {
      int offset = 1 << pass;
      std::optional other = shuffle_down(value, offset, num_threads);
      if((lane_idx + offset < num_values) and other) *value = binary_op(*value, *other);
    }
  }

  return (lane_idx == 0) ? value : std::nullopt;
}



// assume num_values <= warp_size
// the result is returned only to lane 0 if num_values > 0
// the result is undefined for other lanes
template<plain_old_data T, std::invocable<T,T> F>
  requires plain_old_reducible<T*,T*,F>
T warp_reduce(int lane_idx, T value, int num_values, F binary_op)
{
  return subwarp_reduce<warp_size>(lane_idx, value, num_values, binary_op);
}


// assume num_values <= warp_size
// std::nullopt is returned to all other lanes
template<plain_old_data T, std::invocable<T,T> F>
  requires plain_old_reducible<T*,T*,F>
std::optional<T> warp_reduce(int lane_idx, const std::optional<T>& value, int num_values, F binary_op)
{
  return subwarp_reduce<warp_size>(lane_idx, value, num_values, binary_op);
}


constexpr bool is_multiple_of_warp_size(int x)
{
  return x % warp_size == 0;
}


// assume num_values <= block_size
template<int block_size, plain_old_data T, std::invocable<T,T> F>
  requires (is_multiple_of_warp_size(block_size) and plain_old_reducible<T*,T*,F>)
T block_reduce(int thread_idx, T value, int num_values, F binary_op)
{
  constexpr int num_warps = ceil_div(block_size, warp_size);

  __shared__ T s_partial_results[num_warps];

  int warp_idx = thread_idx / warp_size;
  int lane_idx = thread_idx % warp_size;

  int num_values_in_this_warp = warp_size;
  if((warp_idx + 1) * warp_size > num_values)
  {
    num_values_in_this_warp = num_values - warp_idx * warp_size;
  }

  // each warp computes a partial result
  value = warp_reduce(lane_idx, value, num_values_in_this_warp, binary_op);

  __syncthreads();

  // the warp's first lane stores its partial result
  if(lane_idx == 0 and num_values_in_this_warp > 0)
  {
    s_partial_results[warp_idx] = value;
  }

  __syncthreads();

  // the first warp computes the final result from the partials
  int num_partial_results = ceil_div(num_values, warp_size);
  if(warp_idx == 0)
  {
    value = (thread_idx < num_partial_results) ? s_partial_results[thread_idx] : value;
    value = warp_reduce(lane_idx, value, num_partial_results, binary_op);
  }

  return value;
}


template<int block_size, plain_old_data T, std::invocable<T,T> F>
  requires (is_multiple_of_warp_size(block_size) and plain_old_reducible<T*,T*,F>)
std::optional<T> block_reduce(int thread_idx, std::optional<T> value, int num_values, F binary_op)
{
  constexpr int num_warps = ceil_div(block_size, warp_size);

  __shared__ T s_partial_results[num_warps];

  int warp_idx = thread_idx / warp_size;
  int lane_idx = thread_idx % warp_size;

  int num_values_in_this_warp = warp_size;
  if((warp_idx + 1) * warp_size > num_values)
  {
    num_values_in_this_warp = num_values - warp_idx * warp_size;
  }

  // each warp computes a partial result
  value = warp_reduce(lane_idx, value, num_values_in_this_warp, binary_op);

  __syncthreads();

  // the warp's first lane stores its partial result
  if(lane_idx == 0 and num_values_in_this_warp > 0)
  {
    s_partial_results[warp_idx] = *value;
  }

  __syncthreads();

  // the first warp computes the final result from the partials
  int num_partial_results = ceil_div(num_values, warp_size);
  if(warp_idx == 0)
  {
    value = (thread_idx < num_partial_results) ? s_partial_results[thread_idx] : value;
    value = warp_reduce(lane_idx, value, num_partial_results, binary_op);
  }

  return value;
}


template<class T, class Arg1, class Arg2 = Arg1>
concept binary_invocable = std::invocable<T,Arg1,Arg2>;


template<std::ranges::range R, binary_invocable<std::ranges::range_value_t<R&&>> F>
std::optional<std::ranges::range_value_t<R&&>> reduce(R&& r, F op)
{
  using namespace std::ranges;

  std::optional<range_value_t<R&&>> result;

  auto i = begin(r);
  if(i != end(r)) result = *i;
  for(++i; i != end(r); ++i)
  {
    range_value_t<R&&> value = *i;
    result = op(*result, value);
  }

  return result;
}


template<class T>
concept random_access_view_of_ranges =
  std::ranges::view<T> and
  std::ranges::random_access_range<T> and
  std::ranges::range<std::ranges::range_value_t<T>>
;


template<class T>
using range_of_ranges_value_t = std::ranges::range_value_t<std::ranges::range_value_t<T>>;

template<class T>
using range_of_ranges_iterator_t = std::ranges::iterator_t<std::ranges::range_value_t<T>>;


template<int block_size, random_access_view_of_ranges V, std::random_access_iterator O, binary_invocable<range_of_ranges_value_t<V>> F>
  requires (is_multiple_of_warp_size(block_size) and plain_old_reducible<range_of_ranges_iterator_t<V>, O, F>)
void reduce_tiles_kernel(int block_idx, int thread_idx, V tiles, O output, F op)
{
  using namespace std::ranges;
  using namespace std::views;

  // find this block's tile
  auto my_tile = tiles[block_idx];

  // this thread will stride through the tile
  auto my_values = stride(drop(my_tile, thread_idx), block_size);

  // reduce within the thread
  std::optional sum = reduce(my_values, op);

  // reduce across the block
  sum = block_reduce<block_size>(thread_idx, sum, std::min<int>(size(my_tile), block_size), op);
  
  if(thread_idx == 0 and sum)
  {
    output[block_idx] = *sum;
  }
}

