#pragma once

#include "../../../detail/prologue.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <nv/target>

namespace ubu::cuda::detail
{


// XXX these function bodies should really be guarded with if UBU_TARGET(is_device) { ... }


inline void sync_blocks(std::uint32_t expected_num_blocks, volatile std::uint32_t* arrived_num_blocks)
{
#if defined(__CUDACC__)
  bool is_block_leader = (threadIdx.x + threadIdx.y + threadIdx.z == 0);
  bool is_first_block = (blockIdx.x + blockIdx.y + blockIdx.z == 0);
    
  __syncthreads();
    
  if(is_block_leader)
  {
    auto all_arrived = [](std::uint32_t old_arrived, std::uint32_t current_arrived)
    {
      return (((old_arrived ^ current_arrived) & 0x80000000) != 0);
    };

    std::uint32_t increment = is_first_block ? (0x80000000 - (expected_num_blocks - 1)) : 1;

    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70, (
      // update the barrier with memory order release
      //asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(oldArrive) : "l"((unsigned int*)arrived), "r"(nb) : "memory");
      std::uint32_t old_arrived = std::atomic_ref(*arrived_num_blocks).fetch_add(increment, std::memory_order_release);

      std::uint32_t current_arrived;                                                                                                      
      do
      {
        // poll the barrier with memory order acquire
        // XXX can probably use std::atomic_ref instead of this inline PTX
        asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(current_arrived) : "l"((std::uint32_t*)arrived_num_blocks) : "memory");

        // XXX circle crashes on this alternative to the above:
        //     https://godbolt.org/z/qvMbh7G6G
        //current_arrived = std::atomic_ref(*arrived_num_blocks).load(std::memory_order_acquire);
      }
      while(not all_arrived(old_arrived, current_arrived));                                                                            
    ), (
      // older architectures need to use fences
      __threadfence();

      std::uint32_t old_arrived = std::atomic_ref(*arrived_num_blocks).fetch_add(increment);

      while(not all_arrived(old_arrived, *arrived_num_blocks));

      __threadfence();
    ))
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

inline void sync_blocks()
{
#if defined(__CUDACC__)
  std::uint32_t expected_num_blocks = gridDim.x * gridDim.y * gridDim.z;

  sync_blocks(expected_num_blocks, cooperative_kernel_barrier_counter_ptr());
#else
  assert(false);
#endif
}


} // end ubu::cuda::detail

#include "../../../detail/epilogue.hpp"

