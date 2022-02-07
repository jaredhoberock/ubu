#pragma once

#include "../detail/prologue.hpp"

#include "../coordinate/point.hpp"
#include "../detail/reflection.hpp"
#include "detail/launch_as_cuda_kernel.hpp"
#include "event.hpp"
#include "thread_id.hpp"
#include <concepts>
#include <functional>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<std::invocable<cuda::thread_id> F>
  requires std::is_trivially_copyable_v<F>
struct invoke_with_builtin_cuda_indices
{
  F f;

  void operator()() const
  {
#if defined(__CUDACC__)
    if ASPERA_TARGET(is_device())
    {
      // create a thread_id from the built-in variables
      cuda::thread_id idx{{blockIdx.x, blockIdx.y, blockIdx.z}, {threadIdx.x, threadIdx.y, threadIdx.z}};

      // invoke the function with the id
      std::invoke(f, idx);
    }
#endif
  }
};


} // end detail


namespace cuda
{


class kernel_executor
{
  public:
    using coordinate_type = thread_id;
    using event_type = cuda::event;

    constexpr kernel_executor(int device, cudaStream_t stream, std::size_t dynamic_shared_memory_size)
      : device_{device},
        stream_{stream},
        dynamic_shared_memory_size_{dynamic_shared_memory_size}
    {}

    constexpr kernel_executor(int device, cudaStream_t stream)
      : kernel_executor{device, stream, 0}
    {}

    constexpr kernel_executor()
      : kernel_executor{0, cudaStream_t{}}
    {}

    kernel_executor(const kernel_executor&) = default;

    inline coordinate_type bulk_execution_grid(std::size_t n) const
    {
      int block_size = 128;
      int num_blocks = (n + block_size - 1) / block_size;

      return coordinate_type{{num_blocks, 1, 1}, {block_size, 1, 1}};
    }

    template<std::regular_invocable<coordinate_type> F>
      requires std::is_trivially_copyable_v<F>
    inline event_type bulk_execute(const event& before, coordinate_type shape, F f) const
    {
      // make the stream wait on the before event
      detail::throw_on_error(cudaStreamWaitEvent(stream_, before.native_handle()), "kernel_executor::bulk_execute: CUDA error after cudaStreamWaitEvent");

      // convert the shape to dim3
      dim3 grid_dim{static_cast<unsigned int>(shape.block.x), static_cast<unsigned int>(shape.block.y), static_cast<unsigned int>(shape.block.z)};
      dim3 block_dim{static_cast<unsigned int>(shape.thread.x), static_cast<unsigned int>(shape.thread.y), static_cast<unsigned int>(shape.thread.z)};

      // launch the kernel
      detail::launch_as_cuda_kernel(detail::invoke_with_builtin_cuda_indices<F>{f}, grid_dim, block_dim, dynamic_shared_memory_size_, stream_, device_);

      // return a new event recorded on our stream
      return {stream_};
    }


    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    inline event_type execute_after(const event& before, F f) const
    {
      return bulk_execute(before, coordinate_type{::int3{1,1,1}, ::int3{1,1,1}}, [f](coordinate_type)
      {
        // ignore the incoming coordinate and just invoke the function
        std::invoke(f);
      });
    }

    
    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    inline event_type first_execute(F f) const
    {
      return execute_after(event{stream()}, f);
    }


    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    inline void execute(F f) const
    {
      // just discard the result of first_execute
      first_execute(f);
    }


    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    inline void finally_execute_after(const event& before, F f) const
    {
      // just discard the result of execute_after
      execute_after(before, f);
    }


    auto operator<=>(const kernel_executor&) const = default;

    constexpr cudaStream_t stream() const
    {
      return stream_;
    }

    constexpr int device() const
    {
      return device_;
    }

  private:
    int device_;
    cudaStream_t stream_;
    std::size_t dynamic_shared_memory_size_;
};


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

