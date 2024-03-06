#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../detail/reflection/is_device.hpp"
#include <atomic>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <nv/target>

namespace ubu::cuda::detail
{

template<std::integral T>
  requires (sizeof(T) == sizeof(int))
T load_acquire(volatile T* ptr)
{
  if UBU_TARGET(ubu::detail::is_device())
  {
    T result;
    asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(result) : "l"((T*)ptr) : "memory");
    return result;
  }
  else
  {
    // XXX circle crashes on this alternative to the above:
    //     https://godbolt.org/z/qvMbh7G6G
    return std::atomic_ref(*ptr).load(std::memory_order_acquire);
  }
}

inline void arrive_and_wait(bool is_first_thread, std::uint32_t num_expected_threads, volatile uint32_t* counter_and_generation_ptr)
{
#if defined(__CUDACC__)
  auto all_arrived = [](std::uint32_t old_arrived, std::uint32_t current_arrived)
  {
    return (((old_arrived ^ current_arrived) & 0x80000000) != 0);
  };
  
  std::uint32_t increment = is_first_thread ? (0x80000000 - (num_expected_threads - 1)) : 1;
  
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70, (
    // update the barrier with memory order release
    //asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(oldArrive) : "l"((unsigned int*)arrived), "r"(nb) : "memory");
    std::uint32_t old_arrived = std::atomic_ref(*counter_and_generation_ptr).fetch_add(increment, std::memory_order_release);
  
    std::uint32_t current_arrived;                                                                                                      
    do
    {
      // poll the barrier with memory order acquire
      current_arrived = load_acquire(counter_and_generation_ptr);
    }
    while(not all_arrived(old_arrived, current_arrived));                                                                            
  ), (
    // older architectures need to use fences
    __threadfence();
  
    std::uint32_t old_arrived = std::atomic_ref(*counter_and_generation_ptr).fetch_add(increment);
  
    while(not all_arrived(old_arrived, *counter_and_generation_ptr));
  
    __threadfence();
  ))
#else
  assert(false);
#endif
}


inline void sync_grid(std::uint32_t expected_num_blocks, volatile std::uint32_t* arrived_num_blocks)
{
#if defined(__CUDACC__)
  bool is_block_leader = (threadIdx.x + threadIdx.y + threadIdx.z == 0);
  bool is_first_block = (blockIdx.x + blockIdx.y + blockIdx.z == 0);
    
  __syncthreads();
  if(is_block_leader)
  {
    arrive_and_wait(is_first_block, expected_num_blocks, arrived_num_blocks);
  }
  __syncthreads();
#else
  assert(false);
#endif
}

constexpr std::uint32_t ptx_envreg1()
{
#if defined(__CUDACC__)
  std::uint32_t result;
  asm("mov.u32 %0, %%envreg1;" : "=r"(result));
  return result;
#else
  return 0;
#endif
}

constexpr std::uint32_t ptx_envreg2()
{
#if defined(__CUDACC__)
  std::uint32_t result;
  asm("mov.u32 %0, %%envreg2;" : "=r"(result));
  return result;
#else
  return 0;
#endif
}

inline std::uint32_t* cooperative_kernel_barrier_counter_ptr()
{
  // the address of the workspace created by cudaLaunchCooperativeKernel is stored in envregs 1 & 2
  std::uint64_t addr_hi = ptx_envreg1();
  std::uint64_t addr_lo = ptx_envreg2();
  std::uint64_t workspace_addr = (addr_hi << 32) | addr_lo;

  // the barrier's counter is located at an offset of sizeof(std::uint32_t)
  return reinterpret_cast<std::uint32_t*>(workspace_addr + sizeof(std::uint32_t));
}

inline void sync_grid()
{
#if defined(__CUDACC__)
  std::uint32_t expected_num_blocks = gridDim.x * gridDim.y * gridDim.z;

  sync_grid(expected_num_blocks, cooperative_kernel_barrier_counter_ptr());
#else
  assert(false);
#endif
}


} // end ubu::cuda::detail

#include "../../../../detail/epilogue.hpp"

