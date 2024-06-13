#include <ubu/places/causality/initial_happening.hpp>
#include <ubu/places/memory/allocators/concepts/asynchronous_allocator.hpp>
#include <ubu/platforms/cuda/device_allocator.hpp>

namespace ns = ubu;

void test_concepts()
{
  static_assert(ns::asynchronous_allocator<ns::cuda::device_allocator<int>>);
}

template<class T>
void test_asynchronous_allocation()
{
  using namespace ns::cuda;

  device_allocator<T> alloc;

  {
    // test synchronous allocation and asynchronous deletion

    device_ptr<T> ptr = alloc.allocate(1);

    event ready = ns::initial_happening(alloc);

    alloc.deallocate_after(ready, device_span<T>(ptr, 1));
  }

  {
    // test asynchronous allocation and synchronous deletion

    event ready = ns::initial_happening(alloc);

    auto [e, span] = alloc.allocate_after(std::move(ready), 1);

    e.wait();

    alloc.deallocate(span.data(), span.size());
  }

  {
    // test asynchronous allocation and asynchronous deletion

    event ready = ns::initial_happening(alloc);

    auto [e, span] = alloc.allocate_after(std::move(ready), 1);

    event all_done = alloc.deallocate_after(e, span);

    all_done.wait();
  }

}

void test_device_allocator()
{
  test_concepts();
  test_asynchronous_allocation<char>();
  test_asynchronous_allocation<int>();
}

