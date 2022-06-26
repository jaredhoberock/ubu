#include <ubu/event/first_cause.hpp>
#include <ubu/event/wait.hpp>
#include <ubu/cuda/graph_allocator.hpp>
#include <ubu/memory/allocator/asynchronous_allocator.hpp>

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
//    auto ready = ns::first_cause(alloc);
//
//    auto done = alloc.deallocate_after(ready, ptr, 1);
//
//    ns::wait(done);
//  }

  {
    // test asynchronous allocation and synchronous deletion

    auto ready = ns::first_cause(alloc);

    auto [e, ptr] = alloc.allocate_after(std::move(ready), 1);

    ns::wait(e);

    alloc.deallocate(ptr, 1);
  }

  {
    // test asynchronous allocation and asynchronous deletion

    auto ready = ns::first_cause(alloc);

    auto [e, ptr] = alloc.allocate_after(std::move(ready), 1);

    auto all_done = alloc.deallocate_after(e, ptr, 1);

    ns::wait(all_done);
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

