#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection.hpp"
#include "../../memory/allocator/allocate_and_zero_after.hpp"
#include "../../memory/allocator/deallocate_after.hpp"
#include "../../tensor/coordinate/concepts/congruent.hpp"
#include "../../tensor/coordinate/concepts/coordinate.hpp"
#include "../../tensor/coordinate/concepts/weakly_congruent.hpp"
#include "../../tensor/coordinate/point.hpp"
#include "cooperation.hpp"
#include "detail/default_dynamic_shared_memory_size.hpp"
#include "detail/launch_as_kernel.hpp"
#include "detail/throw_on_error.hpp"
#include "device_allocator.hpp"
#include "event.hpp"
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
struct invoke_with_builtin_cuda_indices
{
  F f;

  void operator()() const
  {
#if defined(__CUDACC__)
    if UBU_TARGET(ubu::detail::is_device())
    {
      // create a thread_id from the built-in variables
      thread_id idx{{threadIdx.x, threadIdx.y, threadIdx.z}, {blockIdx.x, blockIdx.y, blockIdx.z}};

      // invoke the function with the id
      std::invoke(f, idx);
    }
#endif
  }
};


} // end detail


class device_executor
{
  public:
    using shape_type = thread_id;
    using happening_type = cuda::event;
    using workspace_type = device_workspace;
    using workspace_shape_type = int2; // XXX ideally, this would simply be grabbed from workspace_type

    constexpr device_executor(int device, cudaStream_t stream, std::size_t dynamic_smem_size)
      : device_{device},
        stream_{stream},
        dynamic_smem_size_{dynamic_smem_size}
    {}

    constexpr device_executor(int device, cudaStream_t stream)
      : device_executor{device, stream, 0}
    {}

    constexpr device_executor()
      : device_executor{0, cudaStream_t{}}
    {}

    device_executor(const device_executor&) = default;

    template<std::regular_invocable<shape_type> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_after(const event& before, shape_type shape, F f) const
    {
      // create the function that will be launched as a kernel
      detail::invoke_with_builtin_cuda_indices<F> kernel{f};

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // launch the kernel
      return detail::launch_as_kernel_after(before, grid_dim, block_dim, dynamic_smem_size_, stream_, device_, kernel);
    }

    // this overload of bulk_execute_after just does a simple conversion of the user's shape type to shape_type
    // and then calls the lower-level function
    // XXX this is the kind of simple adaptation the bulk_execute_after CPO ought to do, but it's tricky to do it in that location atm
    template<std::regular_invocable<int2> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_after(const event& before, int2 shape, F f) const
    {
      // map the int2 to {{thread.x,thread.y,thread.z}, {block.x,block.y,block.z}}
      shape_type native_shape{{shape.x, 1, 1}, {shape.y, 1, 1}};

      return bulk_execute_after(before, native_shape, [f](shape_type native_coord)
      {
        // map the native shape_type back into an int2 and invoke
        std::invoke(f, int2{native_coord.thread.x, native_coord.block.x});
      });
    }

    template<std::regular_invocable<shape_type, workspace_type> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_with_workspace_after(const event& before, shape_type shape, int2 workspace_shape, F f) const
    {
      // decompose workspace shape
      auto [inner_buffer_size, outer_buffer_size] = workspace_shape;

      // get an allocator for the outer workspace buffer
      allocator auto alloc = get_allocator();

      // allocate a zeroed outer buffer after the before event
      auto [outer_buffer_ready, outer_buffer_ptr] = allocate_and_zero_after<std::byte>(alloc, *this, before, outer_buffer_size);

      // launch the kernel after the outer buffer is ready
      std::span<std::byte> outer_buffer(outer_buffer_ptr.to_raw_pointer(), outer_buffer_size);
      event after_kernel = with_dynamic_smem_size(inner_buffer_size).bulk_execute_after(outer_buffer_ready, shape, [=](shape_type coord)
      {
        std::invoke(f, coord, device_workspace(outer_buffer));
      });

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), outer_buffer_ptr, outer_buffer_size);
    }

    // this overload of bulk_execute_with_workspace_after just does a simple conversion of the user's shape type to shape_type
    // and then calls the lower-level function
    // XXX this is the kind of simple adaptation the bulk_execute_with_workspace_after CPO ought to do, but it's tricky to do it in that location atm
    template<std::regular_invocable<int2, workspace_type> F>
      requires std::is_trivially_copyable_v<F>
    inline event bulk_execute_with_workspace_after(const event& before, int2 shape, int2 workspace_shape, F f) const
    {
      // map the int2 to {{thread.x,thread.y,thread.z}, {block.x,block.y,block.z}}
      shape_type native_shape{{shape.x, 1, 1}, {shape.y, 1, 1}};

      return bulk_execute_with_workspace_after(before, native_shape, workspace_shape, [f](shape_type native_coord, workspace_type ws)
      {
        // map the native shape_type back into an int2 and invoke
        std::invoke(f, int2{native_coord.thread.x, native_coord.block.x}, ws);
      });
    }

    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    inline event execute_after(const event& before, F f) const
    {
      return bulk_execute_after(before, shape_type{int3{1,1,1}, int3{1,1,1}}, [f](shape_type)
      {
        // ignore the incoming parameter and just invoke the function
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

    constexpr std::size_t dynamic_smem_size() const
    {
      return dynamic_smem_size_;
    }

    constexpr void dynamic_smem_size(std::size_t num_bytes)
    {
      dynamic_smem_size_ = num_bytes;
    }

  private:
    constexpr device_executor with_dynamic_smem_size(std::size_t dynamic_smem_size) const
    {
      return {device(), stream(), dynamic_smem_size};
    }

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
    std::size_t dynamic_smem_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

