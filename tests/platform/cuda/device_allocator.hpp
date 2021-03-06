#include <ubu/causality/first_cause.hpp>
#include <ubu/memory/allocator/asynchronous_allocator.hpp>
#include <ubu/platform/cuda/device_allocator.hpp>

namespace ns = ubu;

void test_concepts()
{
  static_assert(ns::asynchronous_allocator<ns::cuda::device_allocator<int>>);
}

template<class T>
void test_asynchronous_allocation()
{
  using namespace ns::cuda;

  cudaStream_t stream = 0;
  kernel_executor ex{0,stream};
  device_allocator<T> alloc{ex.device()};

  {
    // test synchronous allocation and asynchronous deletion

    device_ptr<T> ptr = alloc.allocate(1);

    event ready = ns::first_cause(ex);

    alloc.deallocate_after(ready, ptr, 1);
  }

  {
    // test asynchronous allocation and synchronous deletion

    event ready = ns::first_cause(ex);

    auto [e, ptr] = alloc.allocate_after(std::move(ready), 1);

    e.wait();

    alloc.deallocate(ptr, 1);
  }

  {
    // test asynchronous allocation and asynchronous deletion

    event ready = ns::first_cause(ex);

    auto [e, ptr] = alloc.allocate_after(std::move(ready), 1);

    event all_done = alloc.deallocate_after(e, ptr, 1);

    all_done.wait();
  }

}

void test_device_allocator()
{
  test_concepts();
  test_asynchronous_allocation<char>();
  test_asynchronous_allocation<int>();
}

