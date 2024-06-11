#include <ubu/platforms/cuda/device_memory_resource.hpp>
#include <ubu/platforms/cuda/device_ptr.hpp>

#undef NDEBUG
#include <cassert>

#include <concepts>
#include <cuda_runtime_api.h>

namespace ns = ubu;


void test_concepts()
{
  static_assert(std::input_or_output_iterator<ns::cuda::device_ptr<int>>);
  static_assert(std::indirectly_readable<ns::cuda::device_ptr<int>>);
  static_assert(std::forward_iterator<ns::cuda::device_ptr<int>>);
  static_assert(std::bidirectional_iterator<ns::cuda::device_ptr<int>>);
  static_assert(std::random_access_iterator<ns::cuda::device_ptr<int>>);
}


void test_constructors()
{
  using namespace ns::cuda;

  {
    // test default construction
    device_ptr<int> ptr;

    // silence "declared but never referenced" warnings
    static_cast<void>(ptr);
  }

  {
    // test construction from nullptr
    device_ptr<int> ptr{nullptr};

    assert(ptr.to_address() == nullptr);
    assert(!ptr);
  }
}


void test_writeable_device_ptr()
{
  using namespace ns::cuda;

  int* d_array{};
  assert(cudaMalloc(reinterpret_cast<void**>(&d_array), 4 * sizeof(int)) == cudaSuccess);
  int h_array[] = {0, 1, 2, 3};
  
  assert(cudaMemcpy(d_array, h_array, 4 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);
  
  // test construction from raw pointer
  device_ptr<int> ptr{d_array};
  
  // test native_handle
  for(int i = 0; i < 4; ++i)
  {
    assert((ptr + i).to_address() == d_array + i);
  }
  
  // test dereference
  for(int i = 0; i < 4; ++i)
  {
    assert(*(ptr + i) == h_array[i]);
  }
  
  // test subscript
  for(int i = 0; i < 4; ++i)
  {
    assert(ptr[i] == h_array[i]);
  }
  
  // test store
  for(int i = 0; i < 4; ++i)
  {
    ptr[i] = 4 - i;
  }
  
  assert(cudaMemcpy(h_array, d_array, 4 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);
  for(int i = 0; i < 4; ++i)
  {
    assert(h_array[i] == 4 - i);
  }
  
  assert(cudaFree(d_array) == cudaSuccess);
}


void test_readable_device_ptr()
{
  using namespace ns::cuda;

  int* d_array{};
  assert(cudaMalloc(reinterpret_cast<void**>(&d_array), 4 * sizeof(int)) == cudaSuccess);
  int h_array[] = {0, 1, 2, 3};
  
  assert(cudaMemcpy(d_array, h_array, 4 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);
  
  // test construction from raw pointer
  device_ptr<const int> ptr{d_array};
  
  // test native_handle
  for(int i = 0; i < 4; ++i)
  {
    assert((ptr + i).to_address() == d_array + i);
  }
  
  // test dereference
  for(int i = 0; i < 4; ++i)
  {
    assert(*(ptr + i) == h_array[i]);
  }
  
  // test subscript
  for(int i = 0; i < 4; ++i)
  {
    assert(ptr[i] == h_array[i]);
  }
  
  assert(cudaFree(d_array) == cudaSuccess);
}


void test_construct_at()
{
  using namespace ns::cuda;

  int* d_array{};
  assert(cudaMalloc(reinterpret_cast<void**>(&d_array), 4 * sizeof(int)) == cudaSuccess);
  
  device_ptr<int> ptr{d_array};

  for(int i = 0; i < 4; ++i)
  {
    (ptr + i).construct_at(i);
  }

  for(int i = 0; i < 4; ++i)
  {
    assert(i == *(ptr + i));
  }

  assert(cudaFree(d_array) == cudaSuccess);
}


void test_copy_between_devices()
{
  using namespace ns;
  cudaStream_t s = 0;

  cuda::device_memory_resource mem0{0,s}, mem1{1,s};

  // allocate an int on each of two devices
  cuda::device_ptr<int> d_ptr0{reinterpret_cast<int*>(mem0.allocate(sizeof(int))), 0};
  cuda::device_ptr<int> d_ptr1{reinterpret_cast<int*>(mem1.allocate(sizeof(int))), 1};

  *d_ptr0 = 7;
  *d_ptr1 = 13;

  assert(7 == *d_ptr0);
  assert(13 == *d_ptr1);

  {
    // test copy from device 0 to device 1
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr1 = *d_ptr0;

    assert(7 == *d_ptr0);
    assert(7 == *d_ptr1);
    assert(*d_ptr0 == *d_ptr1);

    *d_ptr1 = 13;
  }

  {
    // test copy from device 1 to device 0
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = *d_ptr1;

    assert(13 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = 7;
  }

  mem0.deallocate(d_ptr0.to_address(), sizeof(int));
  mem1.deallocate(d_ptr1.to_address(), sizeof(int));
}


void test_device_ptr()
{
  test_concepts();
  test_constructors();
  test_writeable_device_ptr();
  test_readable_device_ptr();
  test_construct_at();

  int num_devices{};
  assert(cudaGetDeviceCount(&num_devices) == cudaSuccess);

  if(num_devices > 1)
  {
    test_copy_between_devices();
  }
}

