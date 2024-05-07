#pragma once

#include "../../detail/prologue.hpp"

#include "../../memory/allocator/allocate_and_zero_after.hpp"
#include "../../memory/allocator/deallocate_after.hpp"
#include "../../tensor/coordinate/concepts/congruent.hpp"
#include "../../tensor/coordinate/one_extend_coordinate.hpp"
#include "../../tensor/fancy_span.hpp"
#include "cooperation.hpp"
#include "detail/graph_utility_functions.hpp"
#include "device_executor.hpp"
#include "graph_allocator.hpp"
#include "graph_node.hpp"
#include "thread_id.hpp"
#include <concepts>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>


namespace ubu::cuda
{


class graph_executor
{
  public:
    using shape_type = thread_id;
    using happening_type = graph_node;
    using workspace_type = grid_workspace;
    using workspace_shape_type = int2; // XXX ideally, this would simply be grabbed from workspace_type

    constexpr graph_executor(cudaGraph_t graph, int device, cudaStream_t stream, std::size_t dynamic_smem_size)
      : graph_{graph},
        device_{device},
        stream_{stream},
        dynamic_smem_size_{dynamic_smem_size}
    {}

    constexpr graph_executor(cudaGraph_t graph, int device, cudaStream_t stream)
      : graph_executor{graph, device, stream, 0}
    {}

    constexpr graph_executor(cudaGraph_t graph)
      : graph_executor{graph, 0, 0}
    {}

    graph_executor(const graph_executor&) = default;

    constexpr cudaGraph_t graph() const
    {
      return graph_;
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
  
    inline graph_node initial_happening() const
    {
      return {graph(), detail::make_empty_node(graph()), stream()};
    }

    template<congruent<shape_type> S, std::invocable<default_coordinate_t<S>> F>
      requires std::is_trivially_copy_constructible_v<F>
    inline graph_node bulk_execute_after(const graph_node& before, S shape, F f) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_executor::bulk_execute_after: before's graph differs from graph_executor's");
      }

      using user_coord_type = default_coordinate_t<S>;

      // create the function that will be launched as a kernel
      detail::invoke_with_this_thread_id<user_coord_type,F> kernel{f};

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // create the kernel node
      return {graph(), detail::make_kernel_node(graph(), before.native_handle(), grid_dim, block_dim, dynamic_smem_size_, device_, kernel), stream()};
    }

    template<congruent<shape_type> S, congruent<workspace_shape_type> W, std::invocable<default_coordinate_t<S>, workspace_type> F>
      requires std::is_trivially_copy_constructible_v<F>
    inline graph_node bulk_execute_with_workspace_after(const graph_node& before, S shape, W workspace_shape, F f) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_executor::bulk_execute_with_workspace_after: before's graph differs from graph_executor's");
      }

      // decompose workspace shape
      auto [inner_buffer_size, outer_buffer_size] = workspace_shape;

      // get an allocator for the outer workspace buffer
      allocator auto alloc = get_allocator();

      // allocate a zeroed outer buffer after the before node
      auto [outer_buffer_ready, outer_buffer_ptr] = allocate_and_zero_after<std::byte>(alloc, *this, before, outer_buffer_size);

      // launch the kernel after the outer buffer is ready
      std::span<std::byte> outer_buffer(outer_buffer_ptr.to_raw_pointer(), outer_buffer_size);
      graph_node after_kernel = with_dynamic_smem_size(inner_buffer_size).bulk_execute_after(outer_buffer_ready, shape, [=](shape_type coord)
      {
        std::invoke(f, coord, grid_workspace(outer_buffer));
      });

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), fancy_span(outer_buffer_ptr, outer_buffer_size));
    }
  
    template<std::invocable F>
    graph_node execute_after(const graph_node& before, F f) const
    {
      return bulk_execute_after(before, one_extend_coordinate<shape_type>(1), [f](shape_type)
      {
        // ignore the incoming parameter and just invoke the function
        std::invoke(f);
      });
    }
  
    template<std::invocable F>
    graph_node first_execute(F f) const
    {
      return execute_after(initial_happening(), f);
    }
  
    template<std::invocable F>
    void execute(F f) const
    {
      graph_node n = first_execute(f);
      n.wait();
    }
  
    auto operator<=>(const graph_executor&) const = default;

  private:
    constexpr graph_executor with_dynamic_smem_size(std::size_t dynamic_smem_size) const
    {
      return {graph(), device(), stream(), dynamic_smem_size};
    }

    // XXX this is only used to allocate a temporary buffer for bulk_execute_after
    //     so, we should use a different type of allocator optimized for such use
    graph_allocator<std::byte> get_allocator() const
    {
      return {graph_, device_, stream_};
    }

    constexpr static std::pair<dim3,dim3> as_dim3s(shape_type shape)
    {
      dim3 grid_dim{static_cast<unsigned int>(shape.block.x), static_cast<unsigned int>(shape.block.y), static_cast<unsigned int>(shape.block.z)};
      dim3 block_dim{static_cast<unsigned int>(shape.thread.x), static_cast<unsigned int>(shape.thread.y), static_cast<unsigned int>(shape.thread.z)};

      return {block_dim, grid_dim};
    }

    cudaGraph_t graph_;
    int device_;
    cudaStream_t stream_;
    std::size_t dynamic_smem_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

