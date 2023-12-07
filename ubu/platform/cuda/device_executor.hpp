#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection.hpp"
#include "../../grid/coordinate/congruent.hpp"
#include "../../grid/coordinate/coordinate.hpp"
#include "../../grid/coordinate/weakly_congruent.hpp"
#include "../../grid/coordinate/point.hpp"
#include "../../memory/allocator/allocate_and_zero_after.hpp"
#include "../../memory/allocator/deallocate_after.hpp"
#include "detail/default_dynamic_shared_memory_size.hpp"
#include "detail/launch_as_kernel.hpp"
#include "detail/throw_on_error.hpp"
#include "device_allocator.hpp"
#include "event.hpp"
#include "shmalloc.hpp"
#include "thread_id.hpp"
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>

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
      thread_id idx{{threadIdx.x, threadIdx.y, threadIdx.z}, {blockIdx.x, blockIdx.y, blockIdx.z}};

      // invoke the function with the id
      std::invoke(f, idx);
    }
#endif
  }
};

struct workspace_type
{
  // XXX we should use small_span or similar with int size
  std::span<std::byte> buffer;

  struct local_workspace_type
  {
    // XXX we should use small_span or similar with int size
    std::span<std::byte> buffer;

    struct barrier_type
    {
      constexpr void arrive_and_wait() const
      {
#if defined(__CUDACC__)
        __syncthreads();
#endif
      }
    };

    barrier_type barrier;
  };

  local_workspace_type local_workspace;
};

template<std::invocable<thread_id, workspace_type> F>
  requires std::is_trivially_copyable_v<F>
struct invoke_with_builtin_cuda_indices_and_workspace
{
  F f;
  std::span<std::byte> outer_buffer;

  void operator()() const
  {
#if defined(__CUDACC__)
    if UBU_TARGET(ubu::detail::is_device())
    {
      // create a thread_id from the built-in variables
      thread_id idx{{threadIdx.x, threadIdx.y, threadIdx.z}, {blockIdx.x, blockIdx.y, blockIdx.z}};

      // count the number of dynamically-allocated shared memory bytes
      unsigned int dynamic_smem_size;
      asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_smem_size));

      // create workspace
      extern __shared__ std::byte inner_buffer[];
      workspace_type workspace;
      workspace.buffer = outer_buffer;
      workspace.local_workspace.buffer = std::span(inner_buffer, dynamic_smem_size);

      // invoke the function with the id and workspace
      std::invoke(f, idx, workspace);
    }
#endif
  }
};


} // end detail


class device_executor
{
  public:
    constexpr static std::size_t default_on_chip_heap_size = -1;

    using shape_type = thread_id;
    using happening_type = cuda::event;
    using workspace_type = detail::workspace_type;
    using workspace_shape_type = int2; // XXX ideally, this would simply be grabbed from workspace_type

    constexpr device_executor(int device, cudaStream_t stream, std::size_t on_chip_heap_size)
      : device_{device},
        stream_{stream},
        on_chip_heap_size_{on_chip_heap_size}
    {}

    constexpr device_executor(int device, cudaStream_t stream)
      : device_executor{device, stream, default_on_chip_heap_size}
    {}

    constexpr device_executor()
      : device_executor{0, cudaStream_t{}}
    {}

    device_executor(const device_executor&) = default;

    template<std::regular_invocable<shape_type, workspace_type> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_after(const event& before, shape_type shape, int2 workspace_shape, F f) const
    {
      // decompose workspace shape
      auto [inner_buffer_size, outer_buffer_size] = workspace_shape;

      // get an allocator for the outer workspace buffer
      allocator auto alloc = get_allocator();

      // allocate a zeroed outer buffer after the before event
      auto [outer_buffer_ready, outer_buffer_ptr] = allocate_and_zero_after<std::byte>(alloc, *this, before, outer_buffer_size);

      // create the function that will be launched as a kernel
      std::span<std::byte> outer_buffer(outer_buffer_ptr.to_raw_pointer(), outer_buffer_size);
      detail::invoke_with_builtin_cuda_indices_and_workspace<F> kernel{f, outer_buffer};

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // launch the kernel after the outer buffer is ready
      event after_kernel = detail::launch_as_kernel_after(outer_buffer_ready, grid_dim, block_dim, inner_buffer_size, stream_, device_, kernel);

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), outer_buffer_ptr, outer_buffer_size);
    }

    // this overload of bulk_execute_after just does a simple conversion of the user's shape type to shape_type
    // and then calls the lower-level function
    // XXX this is the kind of simple adaptation the bulk_execute_after CPO ought to do, but it's tricky to do it in that location atm
    template<std::regular_invocable<int2, workspace_type> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_after(const event& before, int2 shape, int2 workspace_shape, F f) const
    {
      // map the int2 to {{thread.x,thread.y,thread.z}, {block.x,block.y,block.z}}
      shape_type native_shape{{shape.x, 1, 1}, {shape.y, 1, 1}};

      return bulk_execute_after(before, native_shape, workspace_shape, [f](shape_type native_coord, workspace_type ws)
      {
        // map the native shape_type back into an int2 and invoke
        std::invoke(f, int2{native_coord.thread.x, native_coord.block.x}, ws);
      });
    }

    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    inline event execute_after(const event& before, F f) const
    {
      return bulk_execute_after(before, shape_type{int3{1,1,1}, int3{1,1,1}}, int2(0,0), [f](shape_type, workspace_type)
      {
        // ignore the incoming parameters and just invoke the function
        std::invoke(f);
      });
    }

    auto operator<=>(const device_executor&) const = default;

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
    // XXX this is only used to allocate a temporary buffer for bulk_execute_after
    //     so, we should use a different type of allocator optimized for such use
    device_allocator<std::byte> get_allocator() const
    {
      return {device_, stream_};
    }

    constexpr static std::pair<dim3,dim3> as_dim3s(shape_type shape)
    {
      dim3 grid_dim{static_cast<unsigned int>(shape.block.x), static_cast<unsigned int>(shape.block.y), static_cast<unsigned int>(shape.block.z)};
      dim3 block_dim{static_cast<unsigned int>(shape.thread.x), static_cast<unsigned int>(shape.thread.y), static_cast<unsigned int>(shape.thread.z)};

      return {block_dim, grid_dim};
    }

    int device_;
    cudaStream_t stream_;
    std::size_t on_chip_heap_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

