#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection.hpp"
#include "../../cooperators/workspaces/workspace_shape.hpp"
#include "../../places/memory/allocators/allocate_and_zero_after.hpp"
#include "../../places/memory/allocators/deallocate_after.hpp"
#include "../../tensors/coordinates/concepts/congruent.hpp"
#include "../../tensors/coordinates/coordinate_cast.hpp"
#include "../../tensors/coordinates/one_extend_coordinate.hpp"
#include "../../tensors/coordinates/point.hpp"
#include "../../tensors/coordinates/traits/default_coordinate.hpp"
#include "cooperators.hpp"
#include "detail/launch_as_kernel.hpp"
#include "device_allocator.hpp"
#include "event.hpp"
#include "thread_id.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>

namespace ubu::cuda
{

namespace detail
{


// XXX not sure it's actually good to have this C template parameter
//     alternatively, we'd simply require std::invocable<cuda::thread_id>
//     and have assume f would internally convert to C
template<congruent<thread_id> C, std::invocable<C> F>
  requires std::is_trivially_copy_constructible_v<F>
struct invoke_with_this_thread_id
{
  F f;

  void operator()() const
  {
    std::invoke(f, coordinate_cast<C>(this_thread_id()));
  }
};


} // end detail


class device_executor
{
  public:
    using shape_type = thread_id;
    using happening_type = cuda::event;
    using workspace_type = grid_workspace;
    using workspace_shape_type = workspace_shape_t<workspace_type>;

    constexpr device_executor(int device, cudaStream_t stream, std::size_t dynamic_smem_size)
      : device_{device},
        stream_{stream},
        dynamic_smem_size_{dynamic_smem_size}
    {}

    constexpr device_executor(int device, cudaStream_t stream)
      : device_executor{device, stream, 0}
    {}

    constexpr explicit device_executor(int device)
      : device_executor{device, 0}
    {}

    constexpr device_executor()
      : device_executor{0}
    {}

    device_executor(const device_executor&) = default;

    template<congruent<shape_type> S, std::invocable<default_coordinate_t<S>> F>
      requires std::is_trivially_copy_constructible_v<F>
    inline event bulk_execute_after(const event& before, S shape, F f) const
    {
      using user_coord_type = default_coordinate_t<S>;

      // create the function that will be launched as a kernel
      detail::invoke_with_this_thread_id<user_coord_type,F> kernel{f};

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // launch the kernel
      return detail::launch_as_kernel_after(before, grid_dim, block_dim, dynamic_smem_size_, stream_, device_, kernel);
    }

    template<congruent<shape_type> S, congruent<workspace_shape_type> W, std::invocable<default_coordinate_t<S>, workspace_type> F>
      requires std::is_trivially_copy_constructible_v<F>
    inline event bulk_execute_with_workspace_after(const event& before, S shape, W workspace_shape, F f) const
    {
      // decompose workspace shape
      auto [inner_buffer_size, outer_buffer_size] = workspace_shape;

      // get an allocator for the outer workspace buffer
      allocator auto alloc = get_allocator();

      // allocate a zeroed outer buffer after the before event
      auto [outer_buffer_ready, outer_buffer] = allocate_and_zero_after<std::byte>(alloc, *this, before, outer_buffer_size);

      // launch the kernel after the outer buffer is ready
      std::span raw_outer_buffer(outer_buffer.data().to_raw_pointer(), outer_buffer_size);
      event after_kernel = with_dynamic_smem_size(inner_buffer_size).bulk_execute_after(outer_buffer_ready, shape, [=](auto coord)
      {
        std::invoke(f, coord, grid_workspace(raw_outer_buffer));
      });

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), outer_buffer);
    }

    template<std::invocable F>
      requires std::is_trivially_copy_constructible_v<F>
    inline event execute_after(const event& before, F f) const
    {
      return bulk_execute_after(before, one_extend_coordinate<shape_type>(1), [f](auto)
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
      auto [num_threads, num_blocks] = shape;

      dim3  grid_dim{static_cast<unsigned int>(get<0>(num_blocks)),  static_cast<unsigned int>(get<1>(num_blocks)),  static_cast<unsigned int>(get<2>(num_blocks))};
      dim3 block_dim{static_cast<unsigned int>(get<0>(num_threads)), static_cast<unsigned int>(get<1>(num_threads)), static_cast<unsigned int>(get<2>(num_threads))};

      return {block_dim, grid_dim};
    }

    int device_;
    cudaStream_t stream_;
    std::size_t dynamic_smem_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

