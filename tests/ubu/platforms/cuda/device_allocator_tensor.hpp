#include <cassert>
#include <cstddef>
#include <format>
#include <string>
#include <ubu/places/execution/executors/bulk_execute.hpp>
#include <ubu/places/memory/allocators/allocator_tensor.hpp>
#include <ubu/places/memory/views/memory_view.hpp>
#include <ubu/platforms/cuda/device_allocator.hpp>
#include <ubu/platforms/cuda/device_executor.hpp>

#ifndef __device__
#define __device__
#endif

__device__ int this_device;

int init_this_device()
{
  int num_devices = 0;
  if(cudaError_t e = cudaGetDeviceCount(&num_devices))
  {
    std::string what = std::format("CUDA error after cudaGetDeviceCount: {}\n", cudaGetErrorString(e));
    throw std::runtime_error(what);
  }

  for(int device = 0; device != num_devices; ++device)
  {
    ubu::bulk_execute(ubu::cuda::device_executor(device), 1, [device](auto)
    {
      this_device = device;
    });
  }

  return num_devices;
}


template<class A>
void test_concepts(A)
{
  using namespace ubu;

  static_assert(asynchronous_allocator<A>);
}


template<class T, ubu::coordinate S>
void test_custom_allocate_after(ubu::allocator_tensor<ubu::cuda::device_allocator<T>,S> alloc)
{
  using namespace ubu;

  auto before = initial_happening(alloc);

  // allocate a matrix across devices 0 & 1
  auto [allocated, matrix] = ubu::allocate_after<int>(alloc, before, std::pair(10, 2));

  // fill a column of the matrix on either device
  auto filled_0 = ubu::bulk_execute_after(cuda::device_executor(0), allocated, 10, [=](int row)
  {
    int col = this_device ^ 1;
    matrix[ubu::int2(row,col)] = 13;
  });

  auto filled_1 = ubu::bulk_execute_after(cuda::device_executor(1), allocated, 10, [=](int row)
  {
    int col = this_device ^ 1;
    matrix[ubu::int2(row,col)] = 7;
  });

  auto filled = after_all(filled_0,filled_1);

  // check the entire matrix on both devices
  auto checked_0 = ubu::bulk_execute_after(cuda::device_executor(0), filled, matrix.shape(), [=](ubu::int2 coord)
  {
    auto [row,col] = coord;

    int expected = (col == 0) ? 7 : 13;
    assert(expected == matrix[coord]);
  });

  auto checked_1 = ubu::bulk_execute_after(cuda::device_executor(1), filled, matrix.shape(), [=](ubu::int2 coord)
  {
    auto [row,col] = coord;

    int expected = (col == 0) ? 7 : 13;
    assert(expected == matrix[coord]);
  });

  auto done = after_all(checked_0,checked_1);

  wait(alloc.deallocate_after(done, matrix));
}


template<class T, ubu::coordinate S>
void test_one_extending_default_allocate_after(ubu::allocator_tensor<ubu::cuda::device_allocator<T>,S> alloc)
{
  using namespace ubu;

  auto before = initial_happening(alloc);

  // allocate a vector of doubles
  auto [allocated, vector] = allocate_after<double>(alloc, before, 10);

  static_assert(memory_view_of<decltype(vector),double,int>);
  assert(shape(vector) == 10);

  // fill the tensor on device 0
  auto filled = bulk_execute_after(cuda::device_executor(0), allocated, shape(vector), [=](int coord)
  {
    vector[coord] = 13;
  });

  // check the tensor on both devices
  auto checked_0 = bulk_execute_after(cuda::device_executor(0), filled, shape(vector), [=](int coord)
  {
    assert(13 == vector[coord]);
  });

  auto checked_1 = bulk_execute_after(cuda::device_executor(1), filled, shape(vector), [=](int coord)
  {
    assert(13 == vector[coord]);
  });

  wait(deallocate_after(alloc, after_all(checked_0, checked_1), vector));
}


template<class T, ubu::coordinate S>
void test_allocate_and_zero_after(ubu::allocator_tensor<ubu::cuda::device_allocator<T>,S> alloc)
{
  using namespace ubu;

  auto before = initial_happening(alloc);

  // zero allocate a matrix across devices 0 & 1
  auto [allocated, matrix] = ubu::allocate_and_zero_after<double>(alloc, cuda::device_executor(0), before, std::pair(10, 2));

  // check the entire matrix on both devices
  auto checked_0 = bulk_execute_after(cuda::device_executor(0), allocated, matrix.shape(), [=](ubu::int2 coord)
  {
    assert(0 == matrix[coord]);
  });

  auto checked_1 = bulk_execute_after(cuda::device_executor(1), allocated, matrix.shape(), [=](ubu::int2 coord)
  {
    assert(0 == matrix[coord]);
  });

  auto done = after_all(checked_0,checked_1);

  wait(alloc.deallocate_after(done, matrix));
}


void test_device_allocator_tensor()
{
  try
  {
    // don't test if there's only one GPU
    if(init_this_device() < 2)
    {
      return;
    }

    // create allocators for devices 0 & 1
    ubu::cuda::device_allocator<int> alloc_0(0), alloc_1(1);

    // allow memory access between both devices
    enable_all_access(alloc_0, alloc_1);

    // create a tensor of the allocators
    ubu::allocator_tensor alloc(alloc_0, alloc_1);

    // run the tests
    test_custom_allocate_after(alloc);
    test_one_extending_default_allocate_after(alloc);
    test_allocate_and_zero_after(alloc);
  }
  catch(std::runtime_error&)
  {
    // we expect a non-CUDA compiler to cause all of the above to throw
    // rethrow the exception if this is a CUDA compiler
#ifdef __CUDACC__
    throw;
#endif
  }
}

