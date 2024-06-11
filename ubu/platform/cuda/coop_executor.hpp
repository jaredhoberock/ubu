#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/memory/allocator/allocate_and_zero_after.hpp"
#include "../../places/memory/allocator/deallocate_after.hpp"
#include "../../tensor/coordinate/concepts/congruent.hpp"
#include "../../tensor/coordinate/concepts/coordinate.hpp"
#include "../../tensor/coordinate/coordinate_cast.hpp"
#include "../../tensor/coordinate/point.hpp"
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


template<congruent<ubu::int2> C, std::invocable<C> F>
  requires std::is_trivially_copy_constructible_v<F>
struct invoke_with_int2
{
  F f;

  constexpr void operator()() const
  {
#if defined(__CUDACC__)
    if target(ubu::detail::is_device())
    {
      std::invoke(f, coordinate_cast<C>(ubu::int2(threadIdx.x, blockIdx.x)));
    }
#endif
  }
};


} // end detail


class coop_executor
{
  public:
    using workspace_type = coop_grid_workspace;
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

    template<congruent<shape_type> S, std::invocable<default_coordinate_t<S>> F>
      requires std::is_trivially_copy_constructible_v<F>
    event bulk_execute_after(const event& before, S shape, F f) const
    {
      using user_coord_type = default_coordinate_t<S>;

      // create the function that will be launched as a kernel
      detail::invoke_with_int2<user_coord_type,F> kernel{f};

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // launch the kernel
      return detail::launch_as_cooperative_kernel_after(before, grid_dim, block_dim, dynamic_smem_size_, stream_, device_, kernel);
    }

    template<std::invocable F>
      requires std::is_trivially_copy_constructible_v<F>
    event execute_after(const event& before, F f) const
    {
      return bulk_execute_after(before, ones<shape_type>, [f](shape_type)
      {
        // ignore the incoming parameter and just invoke the function
        std::invoke(f);
      });
    }

    template<congruent<shape_type> S, congruent<workspace_shape_type> W, std::invocable<default_coordinate_t<S>, workspace_type> F>
      requires std::is_trivially_copy_constructible_v<F>
    event bulk_execute_with_workspace_after(const event& before, S shape, W workspace_shape, F f) const
    {
      // decompose workspace shape
      auto [inner_buffer_size, outer_buffer_size] = workspace_shape;

      // get an allocator for the outer workspace buffer
      device_allocator alloc = get_allocator();

      // allocate a zeroed outer buffer after the before event
      auto [outer_buffer_ready, outer_buffer] = allocate_and_zero_after<std::byte>(alloc, *this, before, outer_buffer_size);

      // launch the kernel after the outer buffer is ready
      std::span<std::byte> raw_outer_buffer(outer_buffer.data().to_raw_pointer(), outer_buffer_size);
      event after_kernel = with_dynamic_smem_size(inner_buffer_size).bulk_execute_after(outer_buffer_ready, shape, [=](shape_type coord)
      {
        std::invoke(f, coord, workspace_type(raw_outer_buffer));
      });

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), outer_buffer);
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

