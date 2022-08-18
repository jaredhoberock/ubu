#pragma once

#include "../../detail/prologue.hpp"

#include "../../coordinate/point.hpp"
#include "../../detail/reflection.hpp"
#include "detail/default_dynamic_shared_memory_size.hpp"
#include "detail/launch_as_kernel.hpp"
#include "detail/throw_on_error.hpp"
#include "event.hpp"
#include "shmalloc.hpp"
#include "thread_id.hpp"
#include <bit>
#include <concepts>
#include <functional>

namespace ubu::cuda
{

namespace detail
{


template<std::invocable<thread_id> F>
  requires std::is_trivially_copyable_v<F>
struct init_shmalloc_and_invoke_with_builtin_cuda_indices
{
  F f;
  int on_chip_heap_size;

  void operator()() const
  {
#if defined(__CUDACC__)
    if UBU_TARGET(ubu::detail::is_device())
    {
      // initialize shmalloc
      if(threadIdx.x == 0 and threadIdx.y == 0 and threadIdx.z == 0)
      {
        init_on_chip_malloc(on_chip_heap_size);
      }
      __syncthreads();

      // create a thread_id from the built-in variables
      thread_id idx{{blockIdx.x, blockIdx.y, blockIdx.z}, {threadIdx.x, threadIdx.y, threadIdx.z}};

      // invoke the function with the id
      std::invoke(f, idx);
    }
#endif
  }
};


} // end detail


class kernel_executor
{
  public:
    constexpr static std::size_t default_on_chip_heap_size = -1;

    using coordinate_type = thread_id;
    using happening_type = cuda::event;

    constexpr kernel_executor(int device, cudaStream_t stream, std::size_t on_chip_heap_size)
      : device_{device},
        stream_{stream},
        on_chip_heap_size_{on_chip_heap_size}
    {}

    constexpr kernel_executor(int device, cudaStream_t stream)
      : kernel_executor{device, stream, default_on_chip_heap_size}
    {}

    constexpr kernel_executor()
      : kernel_executor{0, cudaStream_t{}}
    {}

    kernel_executor(const kernel_executor&) = default;

    constexpr static coordinate_type bulk_execution_grid(std::size_t n)
    {
      int block_size = 128;

      // if n happens to be a valid block size, use it for a single block launch
      if(n % 32 == 0 and n <= 1024)
      {
        block_size = n;
      }

      int num_blocks = (n + block_size - 1) / block_size;

      return coordinate_type{{num_blocks, 1, 1}, {block_size, 1, 1}};
    }

    template<std::regular_invocable<coordinate_type> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_after(const event& before, coordinate_type shape, F f) const
    {
      // make the stream wait on the before event
      detail::throw_on_error(cudaStreamWaitEvent(stream_, before.native_handle()), "kernel_executor::bulk_execute_after: CUDA error after cudaStreamWaitEvent");

      // convert the shape to dim3
      dim3 grid_dim{static_cast<unsigned int>(shape.block.x), static_cast<unsigned int>(shape.block.y), static_cast<unsigned int>(shape.block.z)};
      dim3 block_dim{static_cast<unsigned int>(shape.thread.x), static_cast<unsigned int>(shape.thread.y), static_cast<unsigned int>(shape.thread.z)};

      // create the function that will be launched as a kernel
      detail::init_shmalloc_and_invoke_with_builtin_cuda_indices<F> kernel{f,0};

      // compute dynamic_shared_memory_size
      int dynamic_shared_memory_size = (on_chip_heap_size_ == default_on_chip_heap_size) ?
        detail::default_dynamic_shared_memory_size(device_, kernel, block_dim.x * block_dim.y * block_dim.z) :
        on_chip_heap_size_
      ;

      // tell the kernel the size of the on-chip heap
      kernel.on_chip_heap_size = dynamic_shared_memory_size;

      // launch the kernel
      detail::launch_as_kernel(grid_dim, block_dim, dynamic_shared_memory_size, stream_, device_, kernel);

      // return a new event recorded on our stream
      return {device_, stream_};
    }


    template<std::regular_invocable<int2> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_after(const event& before, int2 shape, F f) const
    {
      // map the int2 to {{gx,gy,gz}, {bx, by, bz}}
      coordinate_type native_shape{{shape.x, 1, 1}, {shape.y, 1, 1}};

      return bulk_execute_after(before, native_shape, [f](coordinate_type native_coord)
      {
        // map the native coordinate_type back into an int2 and invoke
        std::invoke(f, int2{native_coord.block.x, native_coord.thread.x});
      });
    }


    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    inline event execute_after(const event& before, F f) const
    {
      return bulk_execute_after(before, coordinate_type{int3{1,1,1}, int3{1,1,1}}, [f](coordinate_type)
      {
        // ignore the incoming coordinate and just invoke the function
        std::invoke(f);
      });
    }

    auto operator<=>(const kernel_executor&) const = default;

    constexpr int device() const
    {
      return device_;
    }

    constexpr cudaStream_t stream() const
    {
      return stream_;
    }

    constexpr std::size_t on_chip_heap_size() const
    {
      return on_chip_heap_size_;
    }

    constexpr void on_chip_heap_size(std::size_t num_bytes)
    {
      on_chip_heap_size_ = num_bytes;
    }

  private:
    int device_;
    cudaStream_t stream_;
    std::size_t on_chip_heap_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

