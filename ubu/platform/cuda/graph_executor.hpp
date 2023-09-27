#pragma once

#include "../../detail/prologue.hpp"

#include "detail/graph_utility_functions.hpp"
#include "device_executor.hpp"
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

    using coordinate_type = thread_id;
    using happening_type = graph_node;

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

    constexpr static coordinate_type bulk_execution_grid(std::size_t n)
    {
      return device_executor::bulk_execution_grid(n);
    }
  
    template<std::invocable<coordinate_type> F>
    graph_node bulk_execute_after(const graph_node& before, coordinate_type shape, F f) const
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
      return bulk_execute_after(before, coordinate_type{ubu::int3{1,1,1}, ubu::int3{1,1,1}}, [f](coordinate_type)
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
    cudaGraph_t graph_;
    int device_;
    cudaStream_t stream_;
    std::size_t on_chip_heap_size_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

