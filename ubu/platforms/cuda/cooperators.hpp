#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection.hpp"
#include "../../places/memory/buffers/basic_buffer.hpp"
#include "cooperators/blocks.hpp"
#include "cooperators/grids.hpp"
#include "cooperators/warps.hpp"
#include "cooperators/detail/sync_grid.hpp"
#include <cstddef>
#include <span>
#include <string_view>


namespace ubu::cuda
{


struct grid_workspace
{
  constexpr static const std::string_view thread_scope = "device";

  basic_buffer<int> buffer;
  block_workspace local_workspace;

  constexpr explicit grid_workspace(std::span<std::byte> outer_buffer = {})
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


struct coop_grid_workspace : grid_workspace
{
  using grid_workspace::grid_workspace;

  struct barrier_type
  {
    constexpr static const std::string_view thread_scope = "device";

    // this limit is imposed by detail::sync_grid_count_half
    constexpr static auto max = std::numeric_limits<std::uint16_t>::max();
  
    inline void arrive_and_wait() const
    {
#if defined(__CUDACC__)
      detail::sync_grid();
#endif
    }
  };

  constexpr static auto max_size = barrier_type::max;
  
  barrier_type barrier;
};


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

