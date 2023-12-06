#pragma once

#include "../../detail/prologue.hpp"

#include "detail/graph_utility_functions.hpp"
#include "device_executor.hpp"
#include "graph_allocator.hpp"
#include "graph_node.hpp"
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
    constexpr static std::size_t default_on_chip_heap_size = -1;

    using shape_type = thread_id;
    using happening_type = graph_node;
    using workspace_type = detail::workspace_type;

    inline graph_executor(cudaGraph_t graph, int device, cudaStream_t stream, std::size_t on_chip_heap_size)
      : graph_{graph},
        device_{device},
        stream_{stream},
        on_chip_heap_size_{on_chip_heap_size}
    {}

    inline graph_executor(cudaGraph_t graph, int device, cudaStream_t stream)
      : graph_executor{graph, device, stream, default_on_chip_heap_size}
    {}

    inline graph_executor(cudaGraph_t graph)
      : graph_executor{graph, 0, 0}
    {}

    graph_executor(const graph_executor&) = default;

    inline cudaGraph_t graph() const
    {
      return graph_;
    }

    inline int device() const
    {
      return device_;
    }

    inline cudaStream_t stream() const
    {
      return stream_;
    }

    inline std::size_t on_chip_heap_size() const
    {
      return on_chip_heap_size_;
    }
  
    inline graph_node first_cause() const
    {
      return {graph(), detail::make_empty_node(graph()), stream()};
    }

    constexpr static shape_type bulk_execution_grid(std::size_t n)
    {
      return device_executor::bulk_execution_grid(n);
    }

    template<std::regular_invocable<shape_type, workspace_type> F>
      requires std::is_trivially_copyable_v<F>
    inline graph_node new_bulk_execute_after(const graph_node& before, shape_type shape, int2 workspace_shape, F f) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_executor::new_bulk_execute_after: before's graph differs from graph_executor's");
      }

      // decompose workspace shape
      auto [inner_buffer_size, outer_buffer_size] = workspace_shape;

      // get an allocator for the outer workspace buffer
      allocator auto alloc = get_allocator();

      // allocate an outer buffer after the before node
      auto [outer_buffer_ready, outer_buffer_ptr] = allocate_after<std::byte>(alloc, before, outer_buffer_size);

      // convert the shape to dim3s
      auto [block_dim, grid_dim] = as_dim3s(shape);

      // create the function that will be launched as a kernel
      std::span<std::byte> outer_buffer(outer_buffer_ptr.to_raw_pointer(), outer_buffer_size);
      detail::invoke_with_builtin_cuda_indices_and_workspace<F> kernel{f, outer_buffer};

      // create the kernel node
      graph_node after_kernel{graph(), detail::make_kernel_node(graph(), outer_buffer_ready.native_handle(), grid_dim, block_dim, inner_buffer_size, device_, kernel), stream()};

      // deallocate outer buffer after the kernel
      return deallocate_after(alloc, std::move(after_kernel), outer_buffer_ptr, outer_buffer_size);
    }

    // this overload of new_bulk_execute_after just does a simple conversion of the user's shape type to shape_type
    // and then calls the lower-level function
    // XXX this is the kind of simple adaptation the new_bulk_execute_after CPO ought to do, but it's tricky to do it in that location atm
    template<std::regular_invocable<int2, workspace_type> F>
      requires std::is_trivially_copyable_v<F>
    inline graph_node new_bulk_execute_after(const graph_node& before, int2 shape, int2 workspace_shape, F f) const
    {
      // map the int2 to {{thread.x,thread.y,thread.z}, {block.x,block.y,block.z}}
      shape_type native_shape{{shape.x, 1, 1}, {shape.y, 1, 1}};

      return new_bulk_execute_after(before, native_shape, workspace_shape, [f](shape_type native_coord, workspace_type ws)
      {
        // map the native shape_type back into an int2 and invoke
        std::invoke(f, int2{native_coord.thread.x, native_coord.block.x}, ws);
      });
    }
  
    template<std::invocable<shape_type> F>
    graph_node bulk_execute_after(const graph_node& before, shape_type shape, F f) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_executor::bulk_execute_after: before's graph differs from graph_executor's");
      }
      
      // convert the shape to dim3
      dim3 grid_dim{static_cast<unsigned int>(shape.block.x), static_cast<unsigned int>(shape.block.y), static_cast<unsigned int>(shape.block.z)};
      dim3 block_dim{static_cast<unsigned int>(shape.thread.x), static_cast<unsigned int>(shape.thread.y), static_cast<unsigned int>(shape.thread.z)}; 

      detail::init_shmalloc_and_invoke_with_builtin_cuda_indices<F> kernel{f,0};

      // compute dynamic_shared_memory_size
      int dynamic_shared_memory_size = (on_chip_heap_size_ == default_on_chip_heap_size) ?
        detail::default_dynamic_shared_memory_size(device_, kernel, block_dim.x * block_dim.y * block_dim.z) :
        on_chip_heap_size_
      ;

      // tell the kernel the size of the on-chip heap
      kernel.on_chip_heap_size = dynamic_shared_memory_size;

      return {graph(), detail::make_kernel_node(graph(), before.native_handle(), grid_dim, block_dim, dynamic_shared_memory_size, device_, kernel), stream()};
    }
  
    template<std::invocable F>
    graph_node execute_after(const graph_node& before, F f) const
    {
      return bulk_execute_after(before, shape_type{ubu::int3{1,1,1}, ubu::int3{1,1,1}}, [f](shape_type)
      {
        // ignore the incoming coordinate and just invoke the function
        std::invoke(f);
      });
    }
  
    template<std::invocable F>
    graph_node first_execute(F f) const
    {
      return execute_after(first_cause(), f);
    }
  
    template<std::invocable F>
    void execute(F f) const
    {
      graph_node n = first_execute(f);
      n.wait();
    }
  
    auto operator<=>(const graph_executor&) const = default;

  private:
    // XXX this is only used to allocate a temporary buffer for new_bulk_execute_after
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
    std::size_t on_chip_heap_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

