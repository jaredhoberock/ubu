#include "validate_workspace.hpp"
#include <cassert>
#include <format>
#include <stdexcept>
#include <string>
#include <ubu/places/execution/executors/executor_tensor.hpp>
#include <ubu/places/execution/executors/bulk_execute.hpp>
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


template<class E>
void test_concepts(E)
{
  using namespace ubu;

  static_assert(executor<E>);

  using A = cuda::device_allocator<std::byte>;
  static_assert(hierarchical_workspace<executor_workspace_t<E,A>>);
}


template<ubu::coordinate S>
void test_bulk_execute(ubu::executor_tensor<ubu::cuda::device_executor,S> exec)
{
  using namespace ubu;

  int num_devices = shape_size(exec.shape());

  bulk_execute(exec, ubu::int2(10,num_devices), [](ubu::int2 coord)
  {
#ifdef __CUDACC__
    assert(10 == blockDim.x);
    assert(1 == blockDim.y * blockDim.z);
    assert(2 == gridDim.x);
    assert(1 == gridDim.y * gridDim.z);

    assert(0 == threadIdx.y);
    assert(0 == threadIdx.z);

    assert(threadIdx.x == coord.x);
    assert(blockIdx.x == coord.y);
    assert(this_device == 0);
#endif
  });

  bulk_execute(exec, ubu::int3(10,1,num_devices), [](ubu::int3 coord)
  {
#ifdef __CUDACC__
    assert(10 == blockDim.x);
    assert(1 == blockDim.y * blockDim.z);
    assert(1 == gridDim.x * gridDim.y * gridDim.z);

    assert(0 == threadIdx.y);
    assert(0 == threadIdx.z);
    
    assert(threadIdx.x == coord.x);
    assert(blockIdx.x  == coord.y);
    assert(this_device == coord.z);
#endif
  });
}


template<ubu::coordinate S>
void test_bulk_execute_with_workspace_after(ubu::executor_tensor<ubu::cuda::device_executor,S> exec)
{
  using namespace ubu;

  int num_devices = shape_size(exec.shape());

  // XXX this allocator should actually correspond to the first device_executor's device
  cuda::device_allocator alloc(0); 

  // we need all other devices to be able to access the workspace allocated by alloc
  //for(int device = 1; device < num_devices; ++device)
  //{
  //  alloc.enable_access_from(device);
  //}

  // XXX i think the above loop is all that is necessary, but a CUDA driver bug ca. 8/2024 requires us
  //     to also give access in the other direction
  enable_all_access(alloc, cuda::device_allocator(1));

  {
    // test bulk_execute_with_workspace_after using exec's shape type

    std::tuple kernel_shape(ubu::int3(4,3,2), ubu::int3(2,3,4), num_devices);

    // this will allocate one int per thread per group
    ubu::int3 workspace_shape(sizeof(int) * shape_size(get<0>(kernel_shape)),
                              sizeof(int) * shape_size(tuples::take<2>(kernel_shape)),
                              sizeof(int) * shape_size(kernel_shape));

    auto before = initial_happening(exec);

    auto after = bulk_execute_with_workspace_after(exec, alloc, before, kernel_shape, workspace_shape, [=](auto coord, auto ws)
    {
#ifdef __CUDACC__
      auto [tid, bid, did] = coord;

      assert(4 == blockDim.x);
      assert(3 == blockDim.y);
      assert(2 == blockDim.z);

      assert(2 == gridDim.x);
      assert(3 == gridDim.y);
      assert(4 == gridDim.z);
      
      assert(threadIdx.x == tid.x);
      assert(threadIdx.y == tid.y);
      assert(threadIdx.z == tid.z);

      assert(blockIdx.x == bid.x);
      assert(blockIdx.y == bid.y);
      assert(blockIdx.z == bid.z);

      assert(this_device == did);
#endif

      validate_workspace(coord, kernel_shape, ws);
    });

    wait(after);
  }

  {
    // test one-extending bulk_execute_with_workspace_after

    ubu::int3 kernel_shape(10, 2, num_devices);

    // this will allocate one int per thread per group
    ubu::int3 workspace_shape(sizeof(int) * kernel_shape[0],
                              sizeof(int) * kernel_shape[0] * kernel_shape[1],
                              sizeof(int) * kernel_shape[0] * kernel_shape[1] * kernel_shape[2]);

    auto before = initial_happening(exec);

    auto after = bulk_execute_with_workspace_after(exec, alloc, before, kernel_shape, workspace_shape, [=](auto coord, auto ws)
    {
#ifdef __CUDACC__
      auto [tid, bid, did] = coord;

      assert(10 == blockDim.x);
      assert(1 == blockDim.y * blockDim.z);
      assert(2 == gridDim.x);
      assert(1 == gridDim.y * gridDim.z);

      assert(0 == threadIdx.y);
      assert(0 == threadIdx.z);

      assert(threadIdx.x == tid);
      assert(blockIdx.x  == bid);
      assert(this_device == did);
#endif

      validate_workspace(coord, kernel_shape, ws);
    });

    wait(after);
  }
}


void test_device_executor_tensor()
{
  try
  {
    // don't test if there's only one GPU
    if(init_this_device() < 2)
    {
      return;
    }
  
    ubu::executor_tensor exec(ubu::cuda::device_executor(0), ubu::cuda::device_executor(1));
  
    test_concepts(exec);
    test_bulk_execute(exec);
    test_bulk_execute_with_workspace_after(exec);
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

