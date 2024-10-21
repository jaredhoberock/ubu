#pragma once

#include "../../../detail/prologue.hpp"

#include "../event.hpp"
#include "has_runtime.hpp"
#include "kernel_for_invocable.hpp"
#include "temporarily_with_current_device.hpp"
#include "throw_on_error.hpp"
#include <concepts>
#include <cstring>
#include <type_traits>


namespace ubu::cuda::detail
{

template<bool cooperative, class A, std::invocable<A> K>
  requires (std::is_pointer_v<K> and std::is_trivially_copy_constructible_v<A>)
event launch_kernel(const event& before, dim3 grid_dim, dim3 block_dim, std::size_t dynamic_shared_memory_size, cudaStream_t stream, int device, K ptr_to_kernel, A arg)
{
#if defined(__CUDACC__)
  void* void_ptr_to_kernel = reinterpret_cast<void*>(ptr_to_kernel);

  temporarily_with_current_device(device, [&]
  {
    // make the stream wait on the before event
    throw_on_error(cudaStreamWaitEvent(stream, before.native_handle()),
      "cuda::detail::launch_kernel: after cudaStreamWaitEvent"
    );

    if UBU_TARGET(has_runtime())
    {
      // ignore empty launches
      if(grid_dim.x * grid_dim.y * grid_dim.z * block_dim.x * block_dim.y * block_dim.z != 0)
      {
        if UBU_TARGET(ubu::detail::is_host())
        {
          // point to the parameter
          void* ptr_to_args[] = {reinterpret_cast<void*>(&arg)};

          if constexpr (cooperative)
          {
            // launch the kernel cooperatively
            throw_on_error(cudaLaunchCooperativeKernel(void_ptr_to_kernel, grid_dim, block_dim, ptr_to_args, dynamic_shared_memory_size, stream),
              "cuda::detail::launch_kernel: after cudaLaunchCooperativeKernel"
            );
          }
          else
          {
            // launch the kernel
            throw_on_error(cudaLaunchKernel(void_ptr_to_kernel, grid_dim, block_dim, ptr_to_args, dynamic_shared_memory_size, stream),
              "cuda::detail::launch_kernel: after cudaLaunchKernel"
            );
          }
        } // is_host
        else
        {
          if constexpr (cooperative)
          {
            ubu::detail::throw_runtime_error("cuda::detail::launch_kernel: cooperative launch is unavailable in device code.");
          }
          else
          {
            // copy the parameter
            void* ptr_to_args = cudaGetParameterBuffer(std::alignment_of_v<A>, sizeof(A));
            std::memcpy(ptr_to_args, &arg, sizeof(A));

            // launch the kernel
            throw_on_error(cudaLaunchDevice(void_ptr_to_kernel, ptr_to_args, grid_dim, block_dim, dynamic_shared_memory_size, stream),
              "cuda::detail::launch_kernel: after cudaLaunchDevice"
            );
          }
        } // is_device
      }
    }
    else
    {
      ubu::detail::throw_runtime_error("cuda::detail::launch_kernel requires the CUDA Runtime.");
    }
  });
#else
  ubu::detail::throw_runtime_error("cuda::detail::launch_kernel requires CUDA C++ language support.");
#endif

  // record an event on the stream
  return {device, stream};
}


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
event launch_as_kernel(const event& before, dim3 grid_dim, dim3 block_dim, std::size_t dynamic_shared_memory_size, cudaStream_t stream, int device, F f)
{
  constexpr bool uncooperative = false;
  return launch_kernel<uncooperative>(before, grid_dim, block_dim, dynamic_shared_memory_size, stream, device, kernel_for_invocable(f), f);
}


template<std::invocable F>
  requires std::is_trivially_copy_constructible_v<F>
event launch_as_cooperative_kernel(const event& before, dim3 grid_dim, dim3 block_dim, std::size_t dynamic_shared_memory_size, cudaStream_t stream, int device, F f)
{
  constexpr bool cooperative = true;
  return launch_kernel<cooperative>(before, grid_dim, block_dim, dynamic_shared_memory_size, stream, device, kernel_for_invocable(f), f);
}


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

