#include <array>
#include <ubu/causality.hpp>
#include <ubu/cooperation/workspace/get_local_workspace.hpp>
#include <ubu/execution/executor/bulk_execute_after.hpp>
#include <ubu/execution/executor/concepts/executor.hpp>
#include <ubu/execution/executor/execute_kernel.hpp>
#include <ubu/execution/executor/first_execute.hpp>
#include <ubu/grid/coordinate/colexicographical_lift.hpp>
#include <ubu/grid/layout/stride/apply_stride.hpp>
#include <ubu/grid/layout/stride/compact_column_major_stride.hpp>
#include <ubu/memory/buffer/reinterpret_buffer.hpp>
#include <ubu/platform/cuda/graph_executor.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


void test_concepts()
{
  static_assert(ns::coordinate<ns::cuda::graph_executor::shape_type>);
  static_assert(std::same_as<ns::cuda::graph_executor::shape_type, ns::executor_coordinate_t<ns::cuda::graph_executor>>);
  static_assert(ns::executor<ns::cuda::graph_executor>);
  static_assert(ns::hierarchical_workspace<ns::executor_workspace_t<ns::cuda::graph_executor>>);
  static_assert(std::same_as<ns::int2, ns::executor_workspace_shape_t<ns::cuda::graph_executor>>);
}


void test_equality(ns::cuda::graph_executor ex1)
{
  using namespace ns;
  
  cuda::graph_executor ex2 = ex1;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


#ifndef __managed__
#define __managed__
#endif


__managed__ int result;
__managed__ int result1;
__managed__ int result2;


void test_first_execute(ns::cuda::graph_executor ex)
{
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


void test_execute_after(ns::cuda::graph_executor ex)
{
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


int hash(ns::cuda::thread_id coord)
{
  return coord.block.x ^ coord.block.y ^ coord.block.z ^ coord.thread.x ^ coord.thread.y ^ coord.thread.z;
}


// this array has 3 + 3 axes to match blockIdx + threadIdx
constexpr ns::int6 array_shape = {2, 4, 6, 8, 10, 12};
__managed__ int array[2][4][6][8][10][12] = {};


void test_bulk_execute_after_member_function(ns::cuda::graph_executor ex)
{
  using namespace ns;

  // partition the array shape into the device_executor's shape type
  // such that nearby threads touch nearby addresses
  
  cuda::device_executor::shape_type shape
  {
    // (thread.x, thread.y, thread.z)
    {array_shape[5],array_shape[4],array_shape[3]},
    // (block.x, block.y, block.z)
    {array_shape[2],array_shape[1],array_shape[0]}
  };

  // create local workspaces with block_size ints and a global workspace with 32 bytes
  ns::int2 workspace_shape(sizeof(int) * shape_size(shape.thread), 32);

  try
  {
    auto before = first_cause(ex);

    auto e = ex.bulk_execute_after(before, shape, workspace_shape, [=](cuda::thread_id coord, cuda::graph_executor::workspace_type ws)
    {
      // hash the coordinate and store the result in the array
      int result = hash(coord);
      array[coord.block.z][coord.block.y][coord.block.x][coord.thread.z][coord.thread.y][coord.thread.x] = result;

      // check that the global workspace works
      std::span<std::byte> global_workspace = get_buffer(ws);

      // each thread checks that the global workspace is initialized to zero
      for(std::byte b : global_workspace)
      {
        assert(b == std::byte(0));
      }

      // check that the local workspace works
      std::span<int> local_indices = reinterpret_buffer<int>(get_buffer(get_local_workspace(ws)));

      // each thread records its local index in the local workspace
      int local_idx = apply_stride(coord.thread, compact_column_major_stride(shape.thread));
      local_indices[local_idx] = local_idx;

      // use the local barrier
      arrive_and_wait(get_barrier(get_local_workspace(ws)));

      // the first thread of the local group checks that each thread was recorded in the local workspace
      if(local_idx == 0)
      {
        int expected = 0;
        for(int idx : local_indices)
        {
          assert(expected == idx);
          ++expected;
        }
      }
    });

    wait(e);

    for(auto coord : lattice(shape))
    {
      int expected = hash(coord);

      int result = array[coord.block.z][coord.block.y][coord.block.x][coord.thread.z][coord.thread.y][coord.thread.x];

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


void test_bulk_execute_after_customization_point(ns::cuda::graph_executor ex)
{
  // XXX at the moment, the only difference between this function and test_bulk_execute_after_member_function
  // is that we provide the shape argument as an int2 and call bulk_execute_after through the CPO
  // in principle, the CPO could do some simple adaptations from the shape parameter type to the executor's
  // native shape type

  using namespace ns;

  // partition the array shape into int2
  ns::int2 shape
  {
    array_shape[5]*array_shape[4]*array_shape[3], // num threads
    array_shape[2]*array_shape[1]*array_shape[0]  // num blocks
  };

  // create a workspace with block_size ints and num_blocks * block_size ints
  ns::int2 workspace_shape(sizeof(int) * shape_size(shape[0]), sizeof(int) * shape_size(shape));

  try
  {
    auto before = first_cause(ex);

    auto e = bulk_execute_after(ex, before, shape, workspace_shape, [=](ns::int2 coord, cuda::graph_executor::workspace_type ws)
    {
      int i = apply_stride(coord, compact_column_major_stride(shape));
      auto c = colexicographical_lift(i, array_shape);

      array[c[0]][c[1]][c[2]][c[3]][c[4]][c[5]] = i;

      // check that the global workspace works
      std::span<std::byte> global_workspace = get_buffer(ws);

      // each thread checks that the global workspace is initialized to zero
      for(std::byte b : global_workspace)
      {
        assert(b == std::byte(0));
      }

      // check that the local workspace works
      std::span<int> local_indices = reinterpret_buffer<int>(get_buffer(get_local_workspace(ws)));

      // each thread records its local index in the local workspace
      int local_idx = apply_stride(coord[0], compact_column_major_stride(shape[0]));
      local_indices[local_idx] = local_idx;

      // use the local barrier
      arrive_and_wait(get_barrier(get_local_workspace(ws)));

      // the first thread of the local group checks that each thread was recorded in the local workspace
      if(local_idx == 0)
      {
        int expected = 0;
        for(int idx : local_indices)
        {
          assert(expected == idx);
          ++expected;
        }
      }
    });

    wait(e);

    for(int i = 0; i < shape_size(array_shape); ++i)
    {
      auto c = colexicographical_lift(i, array_shape);

      assert(i == array[c[0]][c[1]][c[2]][c[3]][c[4]][c[5]]);
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
void test_execute_kernel_customization_point(ns::cuda::graph_executor ex, C shape)
{
  using namespace ns;

  try
  {
    ns::execute_kernel(ex, shape, [=](C coord)
    {
      int i = apply_stride(coord, compact_column_major_stride(shape));
      auto c = colexicographical_lift(i, array_shape);

      array[c[0]][c[1]][c[2]][c[3]][c[4]][c[5]] = i;
    });

    for(int i = 0; i < shape_size(array_shape); ++i)
    {
      auto c = colexicographical_lift(i, array_shape);

      assert(i == array[c[0]][c[1]][c[2]][c[3]][c[4]][c[5]]);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    throw;
#endif
  }
}


void test(ns::cuda::graph_executor ex)
{
  test_equality(ex);
  test_first_execute(ex);
  test_execute_after(ex);

  test_bulk_execute_after_member_function(ex);
  test_bulk_execute_after_customization_point(ex);

  test_execute_kernel_customization_point(ex, array_shape[0]*array_shape[1]*array_shape[2]*array_shape[3]*array_shape[4]*array_shape[5]);
  test_execute_kernel_customization_point(ex, ns::int2{array_shape[0]*array_shape[1]*array_shape[2], array_shape[3]*array_shape[4]*array_shape[5]});
  test_execute_kernel_customization_point(ex, ns::int3{array_shape[0]*array_shape[1], array_shape[2]*array_shape[3], array_shape[4]*array_shape[5]});
  test_execute_kernel_customization_point(ex, ns::int4{array_shape[0]*array_shape[1], array_shape[2]*array_shape[3], array_shape[4], array_shape[5]});
  test_execute_kernel_customization_point(ex, ns::int5{array_shape[0]*array_shape[1], array_shape[2], array_shape[3], array_shape[4], array_shape[5]});
  test_execute_kernel_customization_point(ex, array_shape);
}


void test_with_stream(cudaStream_t s)
{
  cudaGraph_t g{};
  assert(cudaSuccess == cudaGraphCreate(&g,0));

  ns::cuda::graph_executor ex{g, 0, s};

  test(ex);

  assert(cudaSuccess == cudaGraphDestroy(g));
}


void test_with_default_stream()
{
  cudaStream_t s{};
  test_with_stream(s);
}


void test_with_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  test_with_stream(s);

  assert(cudaStreamDestroy(s) == cudaSuccess);
}


void test_graph_executor()
{
  test_concepts();
  test_with_default_stream();
  test_with_new_stream();
}

