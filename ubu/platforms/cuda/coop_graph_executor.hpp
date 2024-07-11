#pragma once

#include "../../detail/prologue.hpp"

#include "../../cooperators/workspaces/workspace_shape.hpp"
#include "../../places/memory/allocators/allocate_and_zero_after.hpp"
#include "../../places/memory/allocators/deallocate_after.hpp"
#include "../../tensors/coordinates/concepts/congruent.hpp"
#include "../../tensors/coordinates/one_extend_coordinate.hpp"
#include "cooperators.hpp"
#include "detail/graph_utility_functions.hpp"
#include "coop_executor.hpp"
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


class coop_graph_executor
{
  public:
    using workspace_type = coop_grid_workspace;
    using workspace_shape_type = workspace_shape_t<workspace_type>;
    using shape_type = workspace_shape_type;
    using happening_type = cuda::event;

    constexpr explicit coop_graph_executor(cudaGraph_t graph, int device = 0, cudaStream_t stream = {}, std::size_t dynamic_smem_size = 0)
      : graph_{graph},
        device_{device},
        stream_{stream},
        dynamic_smem_size_{dynamic_smem_size}
    {}

    coop_graph_executor(const coop_graph_executor&) = default;

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
        throw std::runtime_error("cuda::coop_graph_executor::bulk_execute_after: before's graph differs from graph_executor's");
      }

      using user_coord_type = default_coordinate_t<S>;

      // create the function that will be launched as a kernel
      detail::invoke_with_int2<user_coord_type,F> kernel{f};

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // create the kernel node
      return {graph(), detail::make_cooperative_kernel_node(graph(), before.native_handle(), grid_dim, block_dim, dynamic_smem_size_, device_, kernel), stream()};
    }

    template<congruent<shape_type> S, congruent<workspace_shape_type> W, std::invocable<default_coordinate_t<S>, workspace_type> F>
      requires std::is_trivially_copy_constructible_v<F>
    inline graph_node bulk_execute_with_workspace_after(const graph_node& before, S shape, W workspace_shape, F f) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::coop_graph_executor::bulk_execute_with_workspace_after: before's graph differs from coop_graph_executor's");
      }

      if(shape_size(shape) > workspace_type::max_size)
      {
        throw std::runtime_error("coop_graph_executor::bulk_execute_with_workspace_after: requested shape exceeds capacity of coop_grid_workspace");
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
        std::invoke(f, coord, workspace_type(outer_buffer));
      });

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), outer_buffer_ptr, outer_buffer_size);
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
  
    auto operator<=>(const coop_graph_executor&) const = default;

  private:
    constexpr coop_graph_executor with_dynamic_smem_size(std::size_t dynamic_smem_size) const
    {
      return coop_graph_executor(graph(), device(), stream(), dynamic_smem_size);
    }

    // XXX this is only used to allocate a temporary buffer for bulk_execute_after
    //     so, we should use a different type of allocator optimized for such use
    graph_allocator<std::byte> get_allocator() const
    {
      return {graph_, device_, stream_};
    }

    template<congruent<shape_type> S>
    constexpr static std::pair<dim3,dim3> as_dim3s(S shape)
    {
      dim3 block_dim{static_cast<unsigned int>(get<0>(shape))};
      dim3 grid_dim{static_cast<unsigned int>(get<1>(shape))};

      return {block_dim, grid_dim};
    }

    cudaGraph_t graph_;
    int device_;
    cudaStream_t stream_;
    std::size_t dynamic_smem_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

