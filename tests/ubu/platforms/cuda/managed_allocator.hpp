#include <ubu/places/memory/allocators/concepts/allocator.hpp>
#include <ubu/platforms/cuda/device_executor.hpp>
#include <ubu/platforms/cuda/managed_allocator.hpp>

namespace ns = ubu;

void test_concepts()
{
  static_assert(ns::allocator<ns::cuda::managed_allocator<int>>);
}

template<class T>
void test_allocation()
{
  using namespace ns::cuda;

  cudaStream_t stream = 0;
  device_executor ex{0,stream};
  managed_allocator<T> alloc{ex.device()};

  {
    // test allocation and deletion

    T* ptr = alloc.allocate(1);

    alloc.deallocate(ptr, 1);
  }
}

void test_managed_allocator()
{
  test_concepts();
  test_allocation<char>();
  test_allocation<int>();
}

