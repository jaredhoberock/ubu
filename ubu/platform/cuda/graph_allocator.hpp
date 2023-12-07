#pragma once

#include "../../detail/prologue.hpp"

#include "detail/graph_utility_functions.hpp"
#include "device_allocator.hpp"
#include "device_ptr.hpp"
#include "graph_node.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>


namespace ubu::cuda
{


template<class T>
class graph_allocator
{
  public:
    using value_type = T;
    using pointer = device_ptr<T>;
    using happening_type = graph_node;

    graph_allocator(cudaGraph_t graph, int device, cudaStream_t stream)
      : graph_{graph},
        alloc_{device, stream}
    {}

    graph_allocator(cudaGraph_t graph)
      : graph_allocator{graph, 0, 0}
    {}

    graph_allocator(const graph_allocator&) = default;

    template<class U>
    graph_allocator(const graph_allocator<U>& other)
      : graph_allocator{other.graph(), other.device(), other.stream()}
    {}

    cudaGraph_t graph() const
    {
      return graph_;
    }

    int device() const
    {
      return alloc_.device();
    }

    cudaStream_t stream() const
    {
      return alloc_.stream();
    }

    pointer allocate(std::size_t n) const
    {
      return alloc_.allocate(n);
    }

    void deallocate(pointer ptr, std::size_t n) const
    {
      alloc_.deallocate(ptr,n);
    }

    graph_node first_cause() const
    {
      return {graph(), detail::make_empty_node(graph()), stream()};
    }

    std::pair<graph_node, pointer> allocate_after(const graph_node& before, std::size_t n) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_allocator::allocate_after: before's graph differs from graph_allocator's");
      }

      // handle an empty allocation - CUDA runtime won't accomodate mem alloc node for 0 bytes
      if(n == 0)
      {
        auto node = detail::make_empty_node(graph(), before.native_handle());
        return {graph_node{graph(), node, stream()}, nullptr};
      }

      auto [node, ptr] = detail::make_mem_alloc_node(graph(), before.native_handle(), alloc_.device(), sizeof(T) * n);

      pointer d_ptr{reinterpret_cast<T*>(ptr), alloc_.device()};

      return {graph_node{graph(), node, stream()}, d_ptr};
    }

    graph_node deallocate_after(const graph_node& before, pointer ptr, std::size_t n) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_allocator::deallocate_after: before's graph differs from graph_allocator's");
      }

      // handle an empty deallocation
      if(n == 0)
      {
        auto node = detail::make_empty_node(graph(), before.native_handle());
        return {graph(), node, stream()};
      }

      return {graph(), detail::make_mem_free_node(graph(), before.native_handle(), ptr.to_address()), stream()};
    }

    auto operator<=>(const graph_allocator&) const = default;

  private:
    cudaGraph_t graph_;
    device_allocator<T> alloc_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

