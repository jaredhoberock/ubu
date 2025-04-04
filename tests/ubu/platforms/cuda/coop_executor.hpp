#include "validate_workspace.hpp"
#include <array>
#include <ubu/cooperators/workspaces/get_local_workspace.hpp>
#include <ubu/places/causality.hpp>
#include <ubu/places/execution/executors/bulk_execute_after.hpp>
#include <ubu/places/execution/executors/bulk_execute_with_workspace_after.hpp>
#include <ubu/places/execution/executors/concepts/executor.hpp>
#include <ubu/places/execution/executors/execute_kernel.hpp>
#include <ubu/places/execution/executors/finally_execute_after.hpp>
#include <ubu/places/execution/executors/first_execute.hpp>
#include <ubu/places/memory/views/reinterpret_buffer.hpp>
#include <ubu/platforms/cuda/coop_executor.hpp>
#include <ubu/tensors/coordinates/colexicographical_lift.hpp>
#include <ubu/tensors/views/layouts/strides/apply_stride.hpp>
#include <ubu/tensors/views/layouts/strides/compact_left_major_stride.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


void test_concepts()
{
  static_assert(ns::coordinate<ns::cuda::coop_executor::shape_type>);
  static_assert(std::same_as<ns::cuda::coop_executor::shape_type, ns::executor_coordinate_t<ns::cuda::coop_executor>>);
  static_assert(ns::executor<ns::cuda::coop_executor>);
  static_assert(ns::hierarchical_workspace<ns::executor_workspace_t<ns::cuda::coop_executor>>);
  static_assert(std::same_as<ns::int2, ns::executor_workspace_shape_t<ns::cuda::coop_executor>>);
}


void test_equality(ns::cuda::coop_executor ex1)
{
  using namespace ns;
  
  cuda::coop_executor ex2 = ex1;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


#ifndef __managed__
#define __managed__
#endif


__managed__ int result;
__managed__ int result1;
__managed__ int result2;


void test_first_execute(ns::cuda::coop_executor ex)
{
  using namespace ns;

  result = 0;
  int expected = 13;

  try
  {
    auto e = ns::first_execute(ex, [expected]
    {
      result = expected;
    });

    ns::wait(e);
    assert(expected == result);
  }
  catch(std::runtime_error&)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


void test_execute_after(ns::cuda::coop_executor ex)
{
  using namespace ns;

  result1 = 0;
  int expected1 = 13;
  
  try
  {
    auto e1 = ns::first_execute(ex, [expected1]
    {
      result1 = expected1;
    });

    result2 = 0;
    int expected2 = 7;
    auto e2 = ns::execute_after(ex, e1, [expected1,expected2]
    {
      assert(expected1 == result1);
      result2 = expected2;
    });

    ns::wait(e2);
    assert(expected2 == result2);
  }
  catch(std::runtime_error&)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


void test_finally_execute_after(ns::cuda::coop_executor ex)
{
  using namespace ns;

  result1 = 0;
  int expected1 = 13;
  
  try
  {
    auto e1 = ns::first_execute(ex, [expected1]
    {
      result1 = expected1;
    });

    result2 = 0;
    int expected2 = 7;
    ns::finally_execute_after(ex, e1, [expected1,expected2]
    {
      assert(expected1 == result1);
      result2 = expected2;
    });

    assert(cudaStreamSynchronize(ex.stream()) == cudaSuccess);
    assert(expected2 == result2);
  }
  catch(std::runtime_error&)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


int hash(ns::uint2 coord)
{
  return coord.x ^ coord.y;
}


// this array has 2 axes to match threadIdx + blockIdx
constexpr std::array<int,2> array_shape = {2, 4};
__managed__ int array[2][4] = {};


void test_bulk_execute_after_member_function(ns::cuda::coop_executor ex)
{
  using namespace ns;

  // partition the array shape into the coop_executor's shape type
  // such that nearby threads touch nearby addresses
  
  cuda::coop_executor::shape_type shape {array_shape[0], array_shape[1]};

  try
  {
    cuda::event before = initial_happening(ex);

    cuda::event e = ex.bulk_execute_after(before, shape, [=](ns::int2 coord)
    {
      // hash the coordinate and store the result in the array
      array[coord.y][coord.x] = hash(coord);
    });

    wait(e);

    for(auto coord : lattice(shape))
    {
      int expected = hash(coord);

      int result = array[coord.y][coord.x];

      assert(expected == result);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


void test_bulk_execute_after_customization_point(ns::cuda::coop_executor ex)
{
  using namespace ns;

  // partition the array shape into int2
  ns::int2 shape
  {
    array_shape[1], // num threads
    array_shape[0]  // num blocks
  };

  try
  {
    cuda::event before = initial_happening(ex);

    cuda::event e = bulk_execute_after(ex, before, shape, [=](ns::int2 coord)
    {
      int i = apply_stride(compact_left_major_stride(shape), coord);
      auto c = colexicographical_lift(i, array_shape);

      array[c[0]][c[1]] = i;
    });

    wait(e);

    for(int i = 0; i < shape_size(array_shape); ++i)
    {
      auto c = colexicographical_lift(i, array_shape);

      assert(i == array[c[0]][c[1]]);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


void test_bulk_execute_with_workspace_after_member_function(ns::cuda::coop_executor ex)
{
  using namespace ns;

  // partition the array shape into the coop_executor's shape type
  // such that nearby threads touch nearby addresses
  int block_size = array_shape[1];
  int num_blocks = array_shape[0];

  cuda::coop_executor::shape_type shape(block_size, num_blocks);

  // create workspaces with one int per thread per group
  ns::int2 workspace_shape(sizeof(int) * block_size, sizeof(int) * block_size * num_blocks);

  try
  {
    cuda::event before = initial_happening(ex);

    cuda::event e = ex.bulk_execute_with_workspace_after(before, shape, workspace_shape, [=](ns::int2 coord, cuda::coop_executor::workspace_type ws)
    {
      // hash the coordinate and store the result in the array
      int result = hash(coord);
      array[coord.y][coord.x] = result;

      // check that the workspace works
      validate_workspace(coord, shape, ws);
    });

    wait(e);

    for(auto coord : lattice(shape))
    {
      int expected = hash(coord);
      int result = array[coord.y][coord.x];
      assert(expected == result);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


void test_bulk_execute_with_workspace_after_customization_point(ns::cuda::coop_executor ex)
{
  using namespace ns;

  // partition the array shape into the coop_executor's shape type
  // such that nearby threads touch nearby addresses
  int block_size = array_shape[1];
  int num_blocks = array_shape[0];

  cuda::coop_executor::shape_type shape(block_size, num_blocks);

  // create workspaces with one int per thread per group
  ns::int2 workspace_shape(sizeof(int) * block_size, sizeof(int) * block_size * num_blocks);

  try
  {
    ns::cuda::device_allocator<std::byte> alloc;

    cuda::event before = initial_happening(ex);

    cuda::event e = bulk_execute_with_workspace_after(ex, alloc, before, shape, workspace_shape, [=](ns::int2 coord, cuda::coop_executor::workspace_type ws)
    {
      // hash the coordinate and store the result in the array
      int result = hash(coord);
      array[coord.y][coord.x] = result;

      // check that the workspace works
      validate_workspace(coord, shape, ws);
    });

    wait(e);

    for(auto coord : lattice(shape))
    {
      int expected = hash(coord);
      int result = array[coord.y][coord.x];
      assert(expected == result);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


template<ns::coordinate C>
void test_execute_kernel_customization_point(ns::cuda::coop_executor ex, C shape)
{
  using namespace ns;

  try
  {
    ns::execute_kernel(ex, shape, [=](C coord)
    {
      int i = apply_stride(compact_left_major_stride(shape), coord);
      auto c = colexicographical_lift(i, array_shape);

      array[c[0]][c[1]] = i;
    });

    for(int i = 0; i < ns::shape_size(array_shape); ++i)
    {
      auto c = colexicographical_lift(i, array_shape);

      assert(i == array[c[0]][c[1]]);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


void test(ns::cuda::coop_executor ex)
{
  test_equality(ex);
  test_finally_execute_after(ex);
  test_first_execute(ex);
  test_execute_after(ex);

  test_bulk_execute_after_member_function(ex);
  test_bulk_execute_after_customization_point(ex);

  test_bulk_execute_with_workspace_after_member_function(ex);
  test_bulk_execute_with_workspace_after_customization_point(ex);

  test_execute_kernel_customization_point(ex, array_shape[0]*array_shape[1]);
  test_execute_kernel_customization_point(ex, ns::int2{array_shape[0], array_shape[1]});
}


void test_with_default_stream()
{
  cudaStream_t s{};

  ns::cuda::coop_executor ex{0, s};
  test(ex);
}


void test_with_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  ns::cuda::coop_executor ex{0, s};
  test(ex);

  assert(cudaStreamDestroy(s) == cudaSuccess);
}


void test_coop_executor()
{
  test_concepts();
  test_with_default_stream();
  test_with_new_stream();
}

