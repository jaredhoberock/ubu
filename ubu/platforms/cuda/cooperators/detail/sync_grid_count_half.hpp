#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../detail/atomic.hpp"
#include "sync_grid.hpp"
#include <atomic>
#include <nv/target>
#include <type_traits>

namespace ubu::cuda::detail
{

inline uint32_t fetch_add_to_lower_half(std::atomic_ref<uint32_t> number, uint16_t value)
{
  uint32_t expected;
  uint32_t desired;

  do
  {
    expected = number.load(std::memory_order_acq_rel);
    // XXX circle crashes on this alternative to the above:
    //expected = number.load();

    uint16_t lower_half = expected & 0x0000FFFF;
    lower_half += value;
    desired = (expected & 0xFFFF0000) | lower_half;
  }
  while(not number.compare_exchange_weak(expected, desired));

  return expected;
}

inline uint16_t arrive_and_wait_and_sum(uint16_t value, uint32_t num_expected_threads, uint32_t* counter_and_generation_ptr, uint32_t* accumulator_ptr, uint16_t accumulator_reset_value = 0)
{
  using namespace ubu::detail;

  const uint32_t generation_bit = 1 << 31;
  const uint32_t counter_mask = generation_bit - 1;

  // get atomic_refs
  std::atomic_ref counter_and_generation(*counter_and_generation_ptr);
  std::atomic_ref accumulator(*accumulator_ptr);

  // add our value to the accumulator
  fetch_add_to_lower_half(accumulator, value);

  // increment the thread counter
  uint32_t old_counter_and_generation = counter_and_generation.fetch_add(1, std::memory_order_acq_rel);
  bool generation = (old_counter_and_generation & generation_bit) != 0;
  uint32_t old_counter = old_counter_and_generation & counter_mask;

  if(old_counter + 1 == num_expected_threads)
  {
    // the final thread resets state

    // reset the accumulator by moving the lower bits to the high bits
    uint32_t accumulator_value = load_acquire(accumulator_ptr);
    accumulator.store((accumulator_value << 16) | accumulator_reset_value, std::memory_order_relaxed);

    // reset the counter for the next generation
    store_release(counter_and_generation_ptr, generation ? 0 : generation_bit);
  } 
  else
  {
    // all other threads wait for the generation to change
    while((load_acquire(counter_and_generation_ptr) & generation_bit) >> 31 == generation);
  }

  // return the result stored in the high bits of the accumulator
  return (load_acquire(accumulator_ptr) >> 16) - accumulator_reset_value;
}

inline std::uint16_t arrive_and_wait_and_sum(std::uint16_t value)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    const uint32_t num_expected_blocks = gridDim.x * gridDim.y * gridDim.z;

    std::uint32_t* counter_and_generation_ptr = ubu::cuda::detail::cooperative_kernel_barrier_counter_ptr();
    std::uint32_t* accumulator_ptr = counter_and_generation_ptr - 1;

    // cudaLaunchCooperativeKernel initializes the value at grid_workspace->size to 16
    std::uint16_t initial_accumulator_value = 16;

    return arrive_and_wait_and_sum(value, num_expected_blocks, counter_and_generation_ptr, accumulator_ptr, initial_accumulator_value);
  ), (
    assert(false);
    return 0;
  ))
}

inline std::uint16_t sync_grid_count_half(bool predicate)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    std::uint16_t block_count = __syncthreads_count(predicate);

    __shared__ std::uint16_t result;
    if(threadIdx.x == 0)
    {
      result = arrive_and_wait_and_sum(block_count);
    }
    __syncthreads();

    return result;
  ), (
    assert(false);
    return 0;
  ))
}


} // end ubu::cuda::detail

#include "../../../../detail/epilogue.hpp"

