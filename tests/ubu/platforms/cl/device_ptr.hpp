#include <ubu/platforms/cl/context.hpp>
#include <ubu/platforms/cl/device_allocator.hpp>
#include <ubu/platforms/cl/device_ptr.hpp>

#undef NDEBUG
#include <cassert>

#include <concepts>
#include <CL/cl.h>

// XXX these tests are disabled at the moment because
// cl::device_memory_loader is missing upload_after & download_after

//namespace ns = ubu;
//
//
//void test_concepts()
//{
//  static_assert(std::input_or_output_iterator<ns::cl::device_ptr<int>>);
//  static_assert(std::indirectly_readable<ns::cl::device_ptr<int>>);
//  static_assert(std::forward_iterator<ns::cl::device_ptr<int>>);
//  static_assert(std::bidirectional_iterator<ns::cl::device_ptr<int>>);
//  static_assert(std::random_access_iterator<ns::cl::device_ptr<int>>);
//}
//
//
//void test_constructors()
//{
//  using namespace ns::cl;
//
//  {
//    // test default construction
//    device_ptr<int> ptr;
//
//    // silence "declared but never referenced" warnings
//    static_cast<void>(ptr);
//  }
//
//  {
//    // test construction from nullptr
//    device_ptr<int> ptr{nullptr};
//
//    assert(!ptr);
//  }
//}
//
//
//void test_writeable_device_ptr()
//{
//  using namespace ns::cl;
//
//  context ctx;
//
//  cl_int error = 0;
//  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
//  assert(CL_SUCCESS == error);
//
//  cl_mem d_buffer = clCreateBuffer(ctx.native_handle(), CL_MEM_READ_WRITE, 4 * sizeof(int), nullptr, &error);
//  assert(CL_SUCCESS == error);
//
//  int h_array[] = {0, 1, 2, 3};
//  
//  assert(CL_SUCCESS == clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, 4 * sizeof(int), h_array, 0, nullptr, nullptr));
//
//  // test construction
//  device_ptr<int> ptr{{d_buffer, 0}, queue};
//
//  // test dereference
//  for(int i = 0; i < 4; ++i)
//  {
//    assert(*(ptr + i) == h_array[i]);
//  }
//  
//  // test subscript
//  for(int i = 0; i < 4; ++i)
//  {
//    assert(ptr[i] == h_array[i]);
//  }
//  
//  // test store
//  for(int i = 0; i < 4; ++i)
//  {
//    ptr[i] = 4 - i;
//  }
//  
//  assert(CL_SUCCESS == clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, 4 * sizeof(int), h_array, 0, nullptr, nullptr));
//  for(int i = 0; i < 4; ++i)
//  {
//    assert(h_array[i] == 4 - i);
//  }
//
//  assert(CL_SUCCESS == clReleaseMemObject(d_buffer));
//}
//
//
//void test_readable_device_ptr()
//{
//  using namespace ns::cl;
//
//  context ctx;
//
//  cl_int error = 0;
//  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
//  assert(CL_SUCCESS == error);
//
//  cl_mem d_buffer = clCreateBuffer(ctx.native_handle(), CL_MEM_READ_WRITE, 4 * sizeof(int), nullptr, &error);
//  assert(CL_SUCCESS == error);
//
//  int h_array[] = {0, 1, 2, 3};
//  
//  assert(CL_SUCCESS == clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, 4 * sizeof(int), h_array, 0, nullptr, nullptr));
//  
//  // test construction from raw pointer
//  device_ptr<const int> ptr{{d_buffer,0}, queue};
//  
//  // test dereference
//  for(int i = 0; i < 4; ++i)
//  {
//    assert(*(ptr + i) == h_array[i]);
//  }
//  
//  // test subscript
//  for(int i = 0; i < 4; ++i)
//  {
//    assert(ptr[i] == h_array[i]);
//  }
//  
//  assert(CL_SUCCESS == clReleaseMemObject(d_buffer));
//}
//
//
//void test_construct_at()
//{
//  using namespace ns::cl;
//
//  context ctx;
//
//  cl_int error = 0;
//  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
//  assert(CL_SUCCESS == error);
//
//  cl_mem d_buffer = clCreateBuffer(ctx.native_handle(), CL_MEM_READ_WRITE, 4 * sizeof(int), nullptr, &error);
//  assert(CL_SUCCESS == error);
//  
//  device_ptr<int> ptr{{d_buffer, 0}, queue};
//
//  for(int i = 0; i < 4; ++i)
//  {
//    (ptr + i).construct_at(i);
//  }
//
//  for(int i = 0; i < 4; ++i)
//  {
//    assert(i == *(ptr + i));
//  }
//
//  assert(CL_SUCCESS == clReleaseMemObject(d_buffer));
//}
//
//
//void test_copy_between_devices()
//{
//  using namespace ns::cl;
//
//  // create a context
//  context ctx;
//
//  // create a queue for each device
//  cl_int error = 0;
//  cl_command_queue queue0 = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(0), 0, &error);
//  assert(CL_SUCCESS == error);
//
//  cl_command_queue queue1 = clCreateCommandQueueWithProperties(ctx.native_handle(), ctx.device(1), 0, &error);
//  assert(CL_SUCCESS == error);
//
//  // create an allocator for each context
//  device_allocator<int> alloc0{ctx.native_handle(), queue0};
//  device_allocator<int> alloc1{ctx.native_handle(), queue1};
//
//  // allocate an int on each of two devices
//  device_ptr<int> d_ptr0 = alloc0.allocate(1);
//  device_ptr<int> d_ptr1 = alloc1.allocate(1);
//
//  *d_ptr0 = 7;
//  *d_ptr1 = 13;
//
//  assert(7 == *d_ptr0);
//  assert(13 == *d_ptr1);
//
//  {
//    // test copy from device 0 to device 1
//    assert(7 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    *d_ptr1 = *d_ptr0;
//
//    assert(7 == *d_ptr0);
//    assert(7 == *d_ptr1);
//    assert(*d_ptr0 == *d_ptr1);
//
//    *d_ptr1 = 13;
//  }
//
//  {
//    // test copy from device 1 to device 0
//    assert(7 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    *d_ptr0 = *d_ptr1;
//
//    assert(13 == *d_ptr0);
//    assert(13 == *d_ptr1);
//
//    *d_ptr0 = 7;
//  }
//
//  alloc0.deallocate(d_ptr0, 1);
//  alloc1.deallocate(d_ptr1, 1);
//
//  assert(CL_SUCCESS == clReleaseCommandQueue(queue0));
//  assert(CL_SUCCESS == clReleaseCommandQueue(queue1));
//}
//
//
//void test_device_ptr()
//{
//  test_concepts();
//  test_constructors();
//
//  cl_platform_id platform = 0;
//  assert(CL_SUCCESS == clGetPlatformIDs(1, &platform, nullptr));
//
//  cl_uint num_devices{};
//  assert(CL_SUCCESS == clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &num_devices));
//
//  if(num_devices > 0)
//  {
//    test_writeable_device_ptr();
//    test_readable_device_ptr();
//    test_construct_at();
//
//    if(num_devices > 1)
//    {
//      test_copy_between_devices();
//    }
//  }
//}
//

void test_device_ptr()
{
}

