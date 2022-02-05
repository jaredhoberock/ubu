#include <aspera/cuda/device_allocator.hpp>
#include <aspera/memory/allocator/asynchronous_allocator.hpp>

namespace ns = aspera;

void test_concepts()
{
  static_assert(ns::asynchronous_deallocator<ns::cuda::device_allocator<int>>);
  static_assert(ns::asynchronous_allocator<ns::cuda::device_allocator<int>>);
}

template<class T>
void test_asynchronous_allocation()
{
  using namespace ns::cuda;

  cudaStream_t stream = 0;
  kernel_executor ex{stream,0};
  device_allocator<T> alloc{ex.device()};

  {
    // test synchronous allocation and asynchronous deletion

    device_ptr<T> ptr = alloc.allocate(1);

    event ready{ex.device(), stream};

    alloc.deallocate_after(ready, ptr, 1);
  }

  {
    // test asynchronous allocation
    event ready{ex.device(),stream};

    auto future_ptr = alloc.allocate_after(std::move(ready), 1);

    future_ptr.wait();
  }

  {
    // test asynchronous allocation and asynchronous deletion

    event ready{ex.device(),stream};

    auto future_ptr = alloc.allocate_after(std::move(ready), 1);

    auto [e, ptr, n] = std::move(future_ptr).release();

    event all_done = alloc.deallocate_after(e, ptr, n);

    all_done.wait();
  }

  {
    // test asynchronous allocation and synchronous deletion

    event ready{ex.device(),stream};

    auto future_ptr = alloc.allocate_after(std::move(ready), 1);

    future_ptr.wait();

    auto [_, ptr, n] = std::move(future_ptr).release();

    alloc.deallocate(ptr, n);
  }
}

void test_device_allocator()
{
  test_concepts();
  test_asynchronous_allocation<char>();
  test_asynchronous_allocation<int>();
}

