#pragma once

#include "../detail/prologue.hpp"

#include "detail/graph_utility_functions.hpp"
#include "device_allocator.hpp"
#include "device_ptr.hpp"
#include "graph_event.hpp"
#include "graph_executor.hpp"
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
    using event_type = graph_event;

    graph_allocator(cudaGraph_t graph, int device, cudaStream_t stream)
      : graph_{graph},
        alloc_{device, stream}
    {}

    graph_allocator(cudaGraph_t graph)
      : graph_allocator{graph, 0, 0}
    {}

    graph_allocator(const graph_allocator&) = default;

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

    event_type make_independent_event() const
    {
      return {graph(), detail::make_empty_node(graph()), stream()};
    }

    std::pair<event_type, pointer> allocate_after(const event_type& before, std::size_t n) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_allocator::allocate_after: before's graph differs from graph_allocator's");
      }

      auto [node, ptr] = detail::make_mem_alloc_node(graph(), before.native_handle(), alloc_.device(), sizeof(T) * n);

      pointer d_ptr{reinterpret_cast<T*>(ptr), alloc_.device()};

      return {event_type{graph(), node, stream()}, d_ptr};
    }

    event_type deallocate_after(const event_type& before, pointer ptr, std::size_t n) const
    {
      if(before.graph() != graph())
      {
        throw std::runtime_error("cuda::graph_allocator::deallocate_after: before's graph differs from graph_allocator's");
      }

      return {graph(), detail::make_mem_free_node(graph(), before.native_handle(), ptr.to_address()), stream()};
    }

    const graph_executor associated_executor() const
    {
      return {graph(), device(), stream()};
    }

    auto operator<=>(const graph_allocator&) const = default;

  private:
    cudaGraph_t graph_;
    device_allocator<T> alloc_;
};


} // end ubu::cuda


#include "../detail/epilogue.hpp"

