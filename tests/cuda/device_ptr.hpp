#include <aspera/cuda/device_memory_resource.hpp>
#include <aspera/cuda/device_ptr.hpp>
#include <aspera/cuda/kernel_executor.hpp>

#undef NDEBUG
#include <cassert>

#include <cuda_runtime_api.h>

namespace ns = aspera;


void test_copy_n_after()
{
  using namespace ns::cuda;
  cudaStream_t s = 0;

  device_memory_resource mem0{0,s}, mem1{1,s};

  // allocate an int on each of two devices
  device_ptr<int> d_ptr0{reinterpret_cast<int*>(mem0.allocate(sizeof(int))), 0};
  device_ptr<int> d_ptr1{reinterpret_cast<int*>(mem1.allocate(sizeof(int))), 1};

  *d_ptr0 = 7;
  *d_ptr1 = 13;

  assert(7 == *d_ptr0);
  assert(13 == *d_ptr1);

  cudaStream_t s0 = aspera::detail::temporarily_with_current_device(0, [&]
  {
    cudaStream_t result{};
    cudaStreamCreate(&result);
    return result;
  });

  cudaStream_t s1 = aspera::detail::temporarily_with_current_device(1, [&]
  {
    cudaStream_t result{};
    cudaStreamCreate(&result);
    return result;
  });


  {
    // test copy from device 0 to device 1 on device 0
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex0{s0,0};

    event before{ex0.device()};
    auto after = copy_n_after(ex0, before, d_ptr0, 1, d_ptr1);
    after.wait();

    assert(7 == *d_ptr0);
    assert(7 == *d_ptr1);
    assert(*d_ptr0 == *d_ptr1);

    *d_ptr1 = 13;
  }

  {
    // test copy from device 0 to device 1 on device 1
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex1{s1,1};

    event before{ex1.device()};
    auto after = copy_n_after(ex1, before, d_ptr0, 1, d_ptr1);
    after.wait();

    assert(7 == *d_ptr0);
    assert(7 == *d_ptr1);
    assert(*d_ptr0 == *d_ptr1);

    *d_ptr1 = 13;
  }

  {
    // test copy from device 1 to device 0 on device 0
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex0{s0,0};

    event before{ex0.device()};
    auto after = copy_n_after(ex0, before, d_ptr1, 1, d_ptr0);
    after.wait();

    assert(13 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = 7;
  }

  {
    // test copy from device 1 to device 0 on device 1
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex1{s1,1};

    event before{ex1.device()};
    auto after = copy_n_after(ex1, before, d_ptr1, 1, d_ptr0);
    after.wait();

    assert(13 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = 7;
  }


  mem0.deallocate(d_ptr0.native_handle(), sizeof(int));
  mem1.deallocate(d_ptr1.native_handle(), sizeof(int));

  cudaStreamDestroy(s0);
  cudaStreamDestroy(s1);
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

    assert(ptr.native_handle() == nullptr);
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
    assert((ptr + i).native_handle() == d_array + i);
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
    assert((ptr + i).native_handle() == d_array + i);
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


void test_copy_n_after_between_devices()
{
  using namespace ns::cuda;

  cudaStream_t s = 0;
  device_memory_resource mem0{0,s}, mem1{1,s};

  // allocate an int on each of two devices
  device_ptr<int> d_ptr0{reinterpret_cast<int*>(mem0.allocate(sizeof(int))), 0};
  device_ptr<int> d_ptr1{reinterpret_cast<int*>(mem1.allocate(sizeof(int))), 1};

  *d_ptr0 = 7;
  *d_ptr1 = 13;

  assert(7 == *d_ptr0);
  assert(13 == *d_ptr1);

  cudaStream_t s0 = aspera::detail::temporarily_with_current_device(0, [&]
  {
    cudaStream_t result{};
    cudaStreamCreate(&result);
    return result;
  });

  cudaStream_t s1 = aspera::detail::temporarily_with_current_device(1, [&]
  {
    cudaStream_t result{};
    cudaStreamCreate(&result);
    return result;
  });


  {
    // test copy from device 0 to device 1 on device 0
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex0{s0,0};

    event before{ex0.device()};
    auto after = copy_n_after(ex0, before, d_ptr0, 1, d_ptr1);
    after.wait();

    assert(7 == *d_ptr0);
    assert(7 == *d_ptr1);
    assert(*d_ptr0 == *d_ptr1);

    *d_ptr1 = 13;
  }

  {
    // test copy from device 0 to device 1 on device 1
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex1{s1,1};

    event before{ex1.device()};
    auto after = copy_n_after(ex1, before, d_ptr0, 1, d_ptr1);
    after.wait();

    assert(7 == *d_ptr0);
    assert(7 == *d_ptr1);
    assert(*d_ptr0 == *d_ptr1);

    *d_ptr1 = 13;
  }

  {
    // test copy from device 1 to device 0 on device 0
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex0{s0,0};

    event before{ex0.device()};
    auto after = copy_n_after(ex0, before, d_ptr1, 1, d_ptr0);
    after.wait();

    assert(13 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = 7;
  }

  {
    // test copy from device 1 to device 0 on device 1
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    kernel_executor ex1{s1,1};

    event before{ex1.device()};
    auto after = copy_n_after(ex1, before, d_ptr1, 1, d_ptr0);
    after.wait();

    assert(13 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = 7;
  }

  mem0.deallocate(d_ptr0.native_handle(), sizeof(int));
  mem1.deallocate(d_ptr1.native_handle(), sizeof(int));

  assert(cudaStreamDestroy(s0) == cudaSuccess);
  assert(cudaStreamDestroy(s1) == cudaSuccess);
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


void test_device_ptr()
{
  test_constructors();
  test_writeable_device_ptr();
  test_readable_device_ptr();
  test_construct_at();

  int num_devices{};
  assert(cudaGetDeviceCount(&num_devices) == cudaSuccess);

  if(num_devices > 1)
  {
    test_copy_n_after_between_devices();
  }
}

