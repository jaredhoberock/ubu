#include <ubu/cl/context.hpp>
#include <ubu/cl/device_memory_resource.hpp>
#include <ubu/cl/device_ptr.hpp>

#undef NDEBUG
#include <cassert>

#include <CL/cl.h>

namespace ns = ubu;


//void test_copy_n_after()
//{
//  using namespace ns::cuda;
//  cudaStream_t s = 0;
//
//  device_memory_resource mem0{0,s}, mem1{1,s};
//
//  // allocate an int on each of two devices
//  device_ptr<int> d_ptr0{reinterpret_cast<int*>(mem0.allocate(sizeof(int))), 0};
//  device_ptr<int> d_ptr1{reinterpret_cast<int*>(mem1.allocate(sizeof(int))), 1};
//
//  *d_ptr0 = 7;
//  *d_ptr1 = 13;
//
//  assert(7 == *d_ptr0);
//  assert(13 == *d_ptr1);
//
//  cudaStream_t s0 = ubu::detail::temporarily_with_current_device(0, [&]
//  {
//    cudaStream_t result{};
//    cudaStreamCreate(&result);
//    return result;
//  });
//
//  cudaStream_t s1 = ubu::detail::temporarily_with_current_device(1, [&]
//  {
//    cudaStream_t result{};
//    cudaStreamCreate(&result);
//    return result;
//  });
//
//
//  {
//    // test copy from device 0 to device 1 on device 0
//    assert(7 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    kernel_executor ex0{0,s0};
//
//    event before{ex0.stream()};
//    auto after = copy_n_after(ex0, before, d_ptr0, 1, d_ptr1);
//    after.wait();
//
//    assert(7 == *d_ptr0);
//    assert(7 == *d_ptr1);
//    assert(*d_ptr0 == *d_ptr1);
//
//    *d_ptr1 = 13;
//  }
//
//  {
//    // test copy from device 0 to device 1 on device 1
//    assert(7 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    kernel_executor ex1{1,s1};
//
//    event before{ex1.stream()};
//    auto after = copy_n_after(ex1, before, d_ptr0, 1, d_ptr1);
//    after.wait();
//
//    assert(7 == *d_ptr0);
//    assert(7 == *d_ptr1);
//    assert(*d_ptr0 == *d_ptr1);
//
//    *d_ptr1 = 13;
//  }
//
//  {
//    // test copy from device 1 to device 0 on device 0
//    assert(7 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    kernel_executor ex0{0,s0};
//
//    event before{ex0.stream()};
//    auto after = copy_n_after(ex0, before, d_ptr1, 1, d_ptr0);
//    after.wait();
//
//    assert(13 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    *d_ptr0 = 7;
//  }
//
//  {
//    // test copy from device 1 to device 0 on device 1
//    assert(7 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    kernel_executor ex1{1,s1};
//
//    event before{ex1.stream()};
//    auto after = copy_n_after(ex1, before, d_ptr1, 1, d_ptr0);
//    after.wait();
//
//    assert(13 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    *d_ptr0 = 7;
//  }
//
//
//  mem0.deallocate(d_ptr0.native_handle(), sizeof(int));
//  mem1.deallocate(d_ptr1.native_handle(), sizeof(int));
//
//  cudaStreamDestroy(s0);
//  cudaStreamDestroy(s1);
//}


void test_constructors()
{
  using namespace ns::cl;

  {
    // test default construction
    device_ptr<int> ptr;

    // silence "declared but never referenced" warnings
    static_cast<void>(ptr);
  }

  {
    // test construction from nullptr
    device_ptr<int> ptr{nullptr};

    assert(!ptr);
  }
}


void test_writeable_device_ptr()
{
  using namespace ns::cl;

  context ctx;

  cl_int error = 0;
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
  assert(CL_SUCCESS == error);

  cl_mem d_buffer = clCreateBuffer(ctx.native_handle(), CL_MEM_READ_WRITE, 4 * sizeof(int), nullptr, &error);
  assert(CL_SUCCESS == error);

  int h_array[] = {0, 1, 2, 3};
  
  assert(CL_SUCCESS == clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, 4 * sizeof(int), h_array, 0, nullptr, nullptr));

  // test construction
  device_ptr<int> ptr{{d_buffer, 0}, queue};

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
  
  assert(CL_SUCCESS == clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, 4 * sizeof(int), h_array, 0, nullptr, nullptr));
  for(int i = 0; i < 4; ++i)
  {
    assert(h_array[i] == 4 - i);
  }

  assert(CL_SUCCESS == clReleaseMemObject(d_buffer));
}


void test_readable_device_ptr()
{
  using namespace ns::cl;

  context ctx;

  cl_int error = 0;
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
  assert(CL_SUCCESS == error);

  cl_mem d_buffer = clCreateBuffer(ctx.native_handle(), CL_MEM_READ_WRITE, 4 * sizeof(int), nullptr, &error);
  assert(CL_SUCCESS == error);

  int h_array[] = {0, 1, 2, 3};
  
  assert(CL_SUCCESS == clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, 4 * sizeof(int), h_array, 0, nullptr, nullptr));
  
  // test construction from raw pointer
  device_ptr<const int> ptr{{d_buffer,0}, queue};
  
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
  
  assert(CL_SUCCESS == clReleaseMemObject(d_buffer));
}


void test_copy_n()
{
  using namespace ns::cl;

  // create a context
  context ctx;

  // create a queue 
  cl_int error = 0;
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
  assert(CL_SUCCESS == error);

  // create a memory resource
  device_memory_resource mem{ctx.native_handle(), queue};

  // allocate two ints
  device_ptr<int> d_ptr0{mem.allocate(sizeof(int)), queue};
  device_ptr<int> d_ptr1{mem.allocate(sizeof(int)), queue};

  *d_ptr0 = 7;
  *d_ptr1 = 13;

  assert(7 == *d_ptr0);
  assert(13 == *d_ptr1);

  {
    // test copy from 0 to 1
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    copy_n(d_ptr0, 1, d_ptr1);

    assert(7 == *d_ptr0);
    assert(7 == *d_ptr1);
    assert(*d_ptr0 == *d_ptr1);

    *d_ptr1 = 13;
  }

  {
    // test copy from 1 to 0
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    copy_n(d_ptr1, 1, d_ptr0);

    assert(13 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = 7;
  }

  mem.deallocate(d_ptr0.native_handle(), sizeof(int));
  mem.deallocate(d_ptr1.native_handle(), sizeof(int));

  assert(CL_SUCCESS == clReleaseCommandQueue(queue));
}


void test_copy_n_between_devices()
{
  using namespace ns::cl;

  // create a context
  context ctx;

  // create a queue for each device
  cl_int error = 0;
  cl_command_queue queue0 = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
  assert(CL_SUCCESS == error);

  cl_command_queue queue1 = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(1), 0, &error);
  assert(CL_SUCCESS == error);

  // create a memory resource for each context
  device_memory_resource mem0{ctx.native_handle(), queue0};
  device_memory_resource mem1{ctx.native_handle(), queue1};

  // allocate an int on each of two devices
  device_ptr<int> d_ptr0{mem0.allocate(sizeof(int)), queue0};
  device_ptr<int> d_ptr1{mem1.allocate(sizeof(int)), queue1};

  *d_ptr0 = 7;
  *d_ptr1 = 13;

  assert(7 == *d_ptr0);
  assert(13 == *d_ptr1);

  {
    // test copy from device 0 to device 1
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    copy_n(d_ptr0, 1, d_ptr1);

    assert(7 == *d_ptr0);
    assert(7 == *d_ptr1);
    assert(*d_ptr0 == *d_ptr1);

    *d_ptr1 = 13;
  }

  {
    // test copy from device 1 to device 0
    assert(7 == *d_ptr0);
    assert(13 == *d_ptr1);

    copy_n(d_ptr1, 1, d_ptr0);

    assert(13 == *d_ptr0);
    assert(13 == *d_ptr1);

    *d_ptr0 = 7;
  }

  mem0.deallocate(d_ptr0.native_handle(), sizeof(int));
  mem1.deallocate(d_ptr1.native_handle(), sizeof(int));

  assert(CL_SUCCESS == clReleaseCommandQueue(queue0));
  assert(CL_SUCCESS == clReleaseCommandQueue(queue1));
}


void test_construct_at()
{
  using namespace ns::cl;

  context ctx;

  cl_int error = 0;
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
  assert(CL_SUCCESS == error);

  cl_mem d_buffer = clCreateBuffer(ctx.native_handle(), CL_MEM_READ_WRITE, 4 * sizeof(int), nullptr, &error);
  assert(CL_SUCCESS == error);
  
  device_ptr<int> ptr{{d_buffer, 0}, queue};

  for(int i = 0; i < 4; ++i)
  {
    (ptr + i).construct_at(i);
  }

  for(int i = 0; i < 4; ++i)
  {
    assert(i == *(ptr + i));
  }

  assert(CL_SUCCESS == clReleaseMemObject(d_buffer));
}


void test_device_ptr()
{
  test_constructors();

  cl_platform_id platform = 0;
  assert(CL_SUCCESS == clGetPlatformIDs(1, &platform, nullptr));

  cl_uint num_devices{};
  assert(CL_SUCCESS == clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &num_devices));

  if(num_devices > 0)
  {
    test_writeable_device_ptr();
    test_readable_device_ptr();
    test_construct_at();
    test_copy_n();

    if(num_devices > 1)
    {
      test_copy_n_between_devices();
    }
  }
}

