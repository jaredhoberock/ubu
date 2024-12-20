#include <ubu/places/causality/initial_happening.hpp>
#include <ubu/places/causality/wait.hpp>
#include <ubu/places/memory/allocators/concepts/asynchronous_allocator.hpp>
#include <ubu/platforms/cuda/graph_allocator.hpp>

namespace ns = ubu;

void test_concepts()
{
  static_assert(ns::asynchronous_allocator<ns::cuda::graph_allocator<int>>);
}

template<class T>
void test_asynchronous_allocation(ns::cuda::graph_allocator<T> alloc)
{
  using namespace ns;

//  {
//    // XXX CUDA graphs currently (CUDA 11.7) prohibits a graph node from freeing memory that did not originate from an alloc node
//   
//    // test synchronous allocation and asynchronous deletion
//
//    cuda::device_ptr<T> ptr = alloc.allocate(1);
//
//    auto ready = ns::initial_happening(alloc);
//
//    auto done = alloc.deallocate_after(ready, cuda::device_span<T>(ptr, 1));
//
//    ns::wait(done);
//  }

  {
    // test asynchronous allocation and synchronous deletion

    auto ready = initial_happening(alloc);

    auto [e, span] = alloc.allocate_after(std::move(ready), 1);

    wait(e);

    alloc.deallocate(span.data(), span.size());
  }

  {
    // test asynchronous allocation and asynchronous deletion

    auto ready = initial_happening(alloc);

    auto [e, span] = alloc.allocate_after(std::move(ready), 1);

    auto all_done = alloc.deallocate_after(e, span);

    wait(all_done);
  }

}

void test_graph_allocator()
{
  test_concepts();

  cudaGraph_t g{};
  assert(cudaSuccess == cudaGraphCreate(&g,0));

  test_asynchronous_allocation(ns::cuda::graph_allocator<char>(g));
  test_asynchronous_allocation(ns::cuda::graph_allocator<int>(g));

  assert(cudaSuccess == cudaGraphDestroy(g));
}

