#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection.hpp"
#include "../../memory/buffer/empty_buffer.hpp"
#include <cstddef>
#include <span>
#include <string_view>

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


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

