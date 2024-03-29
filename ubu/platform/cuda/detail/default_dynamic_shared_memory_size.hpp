#pragma once

#include "../../../detail/prologue.hpp"

#include "kernel_entry_point.hpp"
#include "temporarily_with_current_device.hpp"
#include "throw_on_error.hpp"
#include <concepts>
#include <cuda_runtime.h>
#include <type_traits>


namespace ubu::cuda::detail
{


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
int max_potential_occupancy(int device, F, int num_threads_per_block, std::size_t dynamic_shared_memory_size)
{
#if defined(__CUDACC__)
  return temporarily_with_current_device(device, [=]
  {
    int result = 0;

    // point to the kernel
    const void* ptr_to_kernel = reinterpret_cast<void*>(kernel_entry_point<F>);

    throw_on_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&result, ptr_to_kernel, num_threads_per_block, dynamic_shared_memory_size),
      "cuda::detail::max_potential_occupancy: after cudaOccupancyMaxActiveBlocksPerMultiprocessor"
    );

    return result;
  });
#else
  ubu::detail::throw_runtime_error("cuda::detail::max_potential_occupancy requires CUDA C++ language support.");
  return 0;
#endif
}


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
std::size_t max_dynamic_shared_memory_size(int device, F, int num_blocks_per_multiprocessor, int num_threads_per_block)
{
#if defined(__CUDACC__)
  return temporarily_with_current_device(device, [=]
  {
    std::size_t result = 0;

    // point to the kernel
    const void* ptr_to_kernel = reinterpret_cast<void*>(kernel_entry_point<F>);

    throw_on_error(cudaOccupancyAvailableDynamicSMemPerBlock(&result, ptr_to_kernel, num_blocks_per_multiprocessor, num_threads_per_block),
      "cuda::detail::max_dynamic_shared_memory_per_block: after cudaOccupancyAvailableDynamicSMemPerBlock"
    );

    return result;
  });
#else
  ubu::detail::throw_runtime_error("cuda::detail::max_dynamic_shared_memory_per_block requires CUDA C++ language support.");
  return 0;
#endif
}


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
std::size_t default_dynamic_shared_memory_size(int device, F f, int num_threads_per_block)
{
  // compute the maximum number of active blocks of F with 0 smem
  int num_blocks_per_multiprocessor = max_potential_occupancy(device, f, num_threads_per_block, 0);

  // compute the largest number of bytes of smem at that occupancy
  std::size_t max_size = max_dynamic_shared_memory_size(device, f, num_blocks_per_multiprocessor, num_threads_per_block);

  // defaulting to the largest possible dynamic smem allocation doesn't actually seem to maximize occupancy in practice 
  // the ratio 5/8 was determined empirically on a simple reduction kernel
  return 5 * max_size / 8;
}


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

