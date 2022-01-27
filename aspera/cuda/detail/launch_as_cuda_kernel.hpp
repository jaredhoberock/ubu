#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection.hpp"
#include "../../detail/exception.hpp"
#include "temporarily_with_current_device.hpp"
#include <concepts>
#include <cstring>
#include <type_traits>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<std::invocable F>
  requires std::is_trivially_copyable_v<F>
__global__ void cuda_kernel_entry_point(F f)
{
  f();
}


template<class Arg>
void workaround_unused_variable_warning(Arg&&) noexcept {}


template<std::invocable F>
  requires std::is_trivially_copyable_v<F>
void launch_as_cuda_kernel(F f, dim3 grid_dim, dim3 block_dim, std::size_t dynamic_shared_memory_size, cudaStream_t stream, int device)
{
#if defined(__CUDACC__)
  detail::temporarily_with_current_device(device, [=]() mutable
  {
    // point to the kernel
    void* ptr_to_kernel = reinterpret_cast<void*>(&cuda_kernel_entry_point<F>);

    // reference the kernel to encourage the compiler not to optimize it away
    workaround_unused_variable_warning(ptr_to_kernel);

    if ASPERA_TARGET(has_cuda_runtime())
    {
      // ignore empty launches
      if(grid_dim.x * grid_dim.y * grid_dim.z * block_dim.x * block_dim.y * block_dim.z != 0)
      {
        if ASPERA_TARGET(is_host())
        {
          // point to the parameter
          void* ptr_to_arg[] = {reinterpret_cast<void*>(&f)};

          // launch the kernel
          detail::throw_on_error(cudaLaunchKernel(ptr_to_kernel, grid_dim, block_dim, ptr_to_arg, dynamic_shared_memory_size, stream),
            "detail::launch_as_cuda_kernel: CUDA error after cudaLaunchKernel"
          );
        }
        else
        {
          // copy the parameter
          void* ptr_to_arg = cudaGetParameterBuffer(std::alignment_of_v<F>, sizeof(F));
          std::memcpy(ptr_to_arg, &f, sizeof(F));

          // launch the kernel
          detail::throw_on_error(cudaLaunchDevice(ptr_to_kernel, ptr_to_arg, grid_dim, block_dim, dynamic_shared_memory_size, stream),
            "detail::launch_as_cuda_kernel: CUDA error after cudaLaunchDevice"
          );
        }
      }
    }
    else
    {
      detail::throw_runtime_error("detail::launch_cuda_kernel requires the CUDA Runtime.");
    }
  });
#else
  detail::throw_runtime_error("detail::launch_cuda_kernel requires CUDA C++ language support.");
#endif
}


} // end detail


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../../detail/epilogue.hpp"

