#pragma once

#include "../../../detail/prologue.hpp"

#include "../event.hpp"
#include "kernel_entry_point.hpp"
#include "has_runtime.hpp"
#include "temporarily_with_current_device.hpp"
#include "throw_on_error.hpp"
#include <concepts>
#include <cstring>
#include <type_traits>


namespace ubu::cuda::detail
{


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
event launch_as_kernel_after(const event& before, dim3 grid_dim, dim3 block_dim, std::size_t dynamic_shared_memory_size, cudaStream_t stream, int device, F f)
{
#if defined(__CUDACC__)
  temporarily_with_current_device(device, [&]()
  {
    // make the stream wait on the before event
    throw_on_error(cudaStreamWaitEvent(stream, before.native_handle()),
      "cuda::detail::launch_as_kernel_after: after cudaStreamWaitEvent"
    );

    // point to the kernel
    void* ptr_to_kernel = reinterpret_cast<void*>(&kernel_entry_point<F>);

    if UBU_TARGET(has_runtime())
    {
      // ignore empty launches
      if(grid_dim.x * grid_dim.y * grid_dim.z * block_dim.x * block_dim.y * block_dim.z != 0)
      {
        if UBU_TARGET(ubu::detail::is_host())
        {
          // point to the parameter
          void* ptr_to_arg[] = {reinterpret_cast<void*>(&f)};

          // launch the kernel
          throw_on_error(cudaLaunchKernel(ptr_to_kernel, grid_dim, block_dim, ptr_to_arg, dynamic_shared_memory_size, stream),
            "cuda::detail::launch_as_kernel_after: after cudaLaunchKernel"
          );
        }
        else
        {
          // copy the parameter
          void* ptr_to_arg = cudaGetParameterBuffer(std::alignment_of_v<F>, sizeof(F));
          std::memcpy(ptr_to_arg, &f, sizeof(F));

          // launch the kernel
          throw_on_error(cudaLaunchDevice(ptr_to_kernel, ptr_to_arg, grid_dim, block_dim, dynamic_shared_memory_size, stream),
            "cuda::detail::launch_as_kernel_after: after cudaLaunchDevice"
          );
        }
      }
    }
    else
    {
      ubu::detail::throw_runtime_error("cuda::detail::launch_as_kernel_after requires the CUDA Runtime.");
    }
  });
#else
  ubu::detail::throw_runtime_error("cuda::detail::launch_as_kernel_after requires CUDA C++ language support.");
#endif

  // record an event on the stream
  return {device, stream};
}


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

