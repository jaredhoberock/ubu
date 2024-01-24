#pragma once

#include "../../detail/prologue.hpp"

#include "../../grid/coordinate/concepts/congruent.hpp"
#include "../../grid/coordinate/concepts/coordinate.hpp"
#include "../../grid/coordinate/point.hpp"
#include "../../memory/allocator/allocate_and_zero_after.hpp"
#include "../../memory/allocator/deallocate_after.hpp"
#include "cooperation.hpp"
#include "detail/default_dynamic_shared_memory_size.hpp"
#include "detail/launch_as_cooperative_kernel.hpp"
#include "device_allocator.hpp"
#include "event.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>


namespace ubu::cuda
{
namespace detail
{


template<std::invocable<ubu::int2> F>
  requires std::is_trivially_copyable_v<F>
struct invoke_with_int2
{
  F f;

  constexpr void operator()() const
  {
#if defined(__CUDACC__)
    if target(ubu::detail::is_device())
    {
      std::invoke(f, ubu::int2(threadIdx.x, blockIdx.x));
    }
#endif
  }
};


} // end detail


class coop_executor
{
  public:
    using workspace_type = concurrent_device_workspace;
    using workspace_shape_type = int2; // XXX ideally, this would simply be grabbed from workspace_type
    using shape_type = workspace_shape_type;
    using happening_type = cuda::event;

    constexpr coop_executor(int device, cudaStream_t stream = cudaStream_t{}, std::size_t dynamic_smem_size = 0)
      : device_{device},
        stream_{stream},
        dynamic_smem_size_{dynamic_smem_size}
    {}

    constexpr coop_executor()
      : coop_executor(0)
    {}

    coop_executor(const coop_executor&) = default;

    auto operator<=>(const coop_executor&) const = default;

    template<congruent<shape_type> S, std::regular_invocable<S> F>
      requires std::is_trivially_copyable_v<F>
    event bulk_execute_after(const event& before, S shape, F f) const
    {
      // create the function that will be launched as a kernel
      detail::invoke_with_int2<F> kernel{f};

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // launch the kernel
      return detail::launch_as_cooperative_kernel_after(before, grid_dim, block_dim, dynamic_smem_size_, stream_, device_, kernel);
    }

    template<std::regular_invocable F>
      requires std::is_trivially_copyable_v<F>
    event execute_after(const event& before, F f) const
    {
      return bulk_execute_after(before, ones<shape_type>, [f](shape_type)
      {
        // ignore the incoming parameter and just invoke the function
        std::invoke(f);
      });
    }

    template<congruent<shape_type> S, std::regular_invocable<S, workspace_type> F>
      requires std::is_trivially_copyable_v<F>
    event bulk_execute_with_workspace_after(const event& before, S shape, int2 workspace_shape, F f) const
    {
      // decompose workspace shape
      auto [inner_buffer_size, outer_buffer_size] = workspace_shape;

      // get an allocator for the outer workspace buffer
      device_allocator alloc = get_allocator();

      // allocate a zeroed outer buffer after the before event
      auto [outer_buffer_ready, outer_buffer_ptr] = allocate_and_zero_after<std::byte>(alloc, *this, before, outer_buffer_size);

      // launch the kernel after the outer buffer is ready
      std::span<std::byte> outer_buffer(outer_buffer_ptr.to_raw_pointer(), outer_buffer_size);
      event after_kernel = with_dynamic_smem_size(inner_buffer_size).bulk_execute_after(outer_buffer_ready, shape, [=](shape_type coord)
      {
        std::invoke(f, coord, workspace_type(outer_buffer));
      });

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), outer_buffer_ptr, outer_buffer_size);
    }

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
    constexpr coop_executor with_dynamic_smem_size(std::size_t dynamic_smem_size) const
    {
      return {device(), stream(), dynamic_smem_size};
    }

    // XXX this is only used to allocate a temporary buffer for bulk_execute_with_workspace_after
    //     so, we should use a different type of allocator optimized for such use
    device_allocator<std::byte> get_allocator() const
    {
      return {device_, stream_};
    }

    template<congruent<shape_type> S>
    constexpr static std::pair<dim3,dim3> as_dim3s(S shape)
    {
      dim3 block_dim{static_cast<unsigned int>(get<0>(shape))};
      dim3 grid_dim{static_cast<unsigned int>(get<1>(shape))};

      return {block_dim, grid_dim};
    }

    int device_;
    cudaStream_t stream_;
    std::size_t dynamic_smem_size_;
};


} // end ubu::cuda

#include "../../detail/epilogue.hpp"

