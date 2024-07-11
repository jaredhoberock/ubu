#include <array>
#include <ubu/cooperators/workspaces/get_local_workspace.hpp>
#include <ubu/places/causality.hpp>
#include <ubu/places/execution/executors/bulk_execute_after.hpp>
#include <ubu/places/execution/executors/bulk_execute_with_workspace_after.hpp>
#include <ubu/places/execution/executors/concepts/executor.hpp>
#include <ubu/places/execution/executors/execute_kernel.hpp>
#include <ubu/places/execution/executors/finally_execute_after.hpp>
#include <ubu/places/execution/executors/first_execute.hpp>
#include <ubu/places/memory/buffers/reinterpret_buffer.hpp>
#include <ubu/platforms/cuda/coop_graph_executor.hpp>
#include <ubu/tensors/coordinates/colexicographical_lift.hpp>
#include <ubu/tensors/views/layouts/strides/apply_stride.hpp>
#include <ubu/tensors/views/layouts/strides/compact_left_major_stride.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


void test_concepts()
{
  static_assert(ns::coordinate<ns::cuda::coop_graph_executor::shape_type>);
  static_assert(std::same_as<ns::cuda::coop_graph_executor::shape_type, ns::executor_coordinate_t<ns::cuda::coop_graph_executor>>);
  static_assert(ns::executor<ns::cuda::coop_graph_executor>);
  static_assert(ns::hierarchical_workspace<ns::executor_workspace_t<ns::cuda::coop_graph_executor>>);
  static_assert(std::same_as<ns::int2, ns::executor_workspace_shape_t<ns::cuda::coop_graph_executor>>);
}


void test_equality(ns::cuda::coop_graph_executor ex1)
{
  using namespace ns;
  
  cuda::coop_graph_executor ex2 = ex1;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


#ifndef __managed__
#define __managed__
#endif


__managed__ int result;
__managed__ int result1;
__managed__ int result2;


void test_first_execute(ns::cuda::coop_graph_executor ex)
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


void test_execute_after(ns::cuda::coop_graph_executor ex)
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


int hash(ns::uint2 coord)
{
  return coord.x ^ coord.y;
}


// this array has 2 axes to match threadIdx + blockIdx
constexpr std::array<int,2> array_shape = {2, 4};
__managed__ int array[2][4] = {};


void test_bulk_execute_after_member_function(ns::cuda::coop_graph_executor ex)
{
  using namespace ns;

  // partition the array shape into the coop_graph_executor's shape type
  // such that nearby threads touch nearby addresses
  
  cuda::coop_graph_executor::shape_type shape {array_shape[0], array_shape[1]};

  try
  {
    auto before = initial_happening(ex);

    auto e = ex.bulk_execute_after(before, shape, [=](ns::int2 coord)
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


void test_bulk_execute_after_customization_point(ns::cuda::coop_graph_executor ex)
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
    auto before = initial_happening(ex);

    auto e = bulk_execute_after(ex, before, shape, [=](ns::int2 coord)
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


void test_bulk_execute_with_workspace_after_member_function(ns::cuda::coop_graph_executor ex)
{
  using namespace ns;

  // partition the array shape into the coop_graph_executor's shape type
  // such that nearby threads touch nearby addresses
  int block_size = array_shape[1];
  int num_blocks = array_shape[0];

  cuda::coop_graph_executor::shape_type shape(block_size, num_blocks);

  // create local workspaces with block_size ints and a global workspace with num_blocks ints
  ns::int2 workspace_shape(block_size * sizeof(int), num_blocks * sizeof(int));

  try
  {
    auto before = initial_happening(ex);

    auto e = ex.bulk_execute_with_workspace_after(before, shape, workspace_shape, [=](ns::int2 coord, cuda::coop_graph_executor::workspace_type ws)
    {
      // hash the coordinate and store the result in the array
      int result = hash(coord);
      array[coord.y][coord.x] = result;

      auto [thread_id, block_id] = coord;

      // test the outer workspace
      {
        std::span<int> block_indices = reinterpret_buffer<int>(get_buffer(ws));

        // the leader of each block records its block index in the global workspace
        if(thread_id == 0)
        {
          block_indices[block_id] = block_id;
        }

        // use the global barrier
        arrive_and_wait(get_barrier(ws));

        // the first thread of the first block checks that each block was recorded in the global workspace
        if(thread_id == 0 and block_id == 0)
        {
          int expected = 0;
          for(int id : block_indices)
          {
            assert(expected == id);
            ++expected;
          }
        }
      }

      // test the local workspace
      {
        std::span<int> thread_indices = reinterpret_buffer<int>(get_buffer(get_local_workspace(ws)));

        // each thread records its index in the local workspace
        thread_indices[thread_id] = thread_id;

        // use the local barrier
        arrive_and_wait(get_barrier(get_local_workspace(ws)));

        // the first thread of each block checks that each thread was recorded in the local workspace
        if(thread_id == 0)
        {
          int expected = 0;
          for(int id : thread_indices)
          {
            assert(expected == id);
            ++expected;
          }
        }
      }
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


void test_bulk_execute_with_workspace_after_customization_point(ns::cuda::coop_graph_executor ex)
{
  using namespace ns;

  // partition the array shape into the coop_graph_executor's shape type
  // such that nearby threads touch nearby addresses
  int block_size = array_shape[1];
  int num_blocks = array_shape[0];

  cuda::coop_graph_executor::shape_type shape(block_size, num_blocks);

  // create local workspaces with block_size ints and a global workspace with num_blocks ints
  ns::int2 workspace_shape(block_size * sizeof(int), num_blocks * sizeof(int));

  try
  {
    ns::cuda::graph_allocator<std::byte> alloc{ex.graph(), ex.device(), ex.stream()};

    auto before = initial_happening(ex);

    auto e = bulk_execute_with_workspace_after(ex, alloc, before, shape, workspace_shape, [=](ns::int2 coord, cuda::coop_graph_executor::workspace_type ws)
    {
      // hash the coordinate and store the result in the array
      int result = hash(coord);
      array[coord.y][coord.x] = result;

      auto [thread_id, block_id] = coord;

      // test the outer workspace
      {
        std::span<int> block_indices = reinterpret_buffer<int>(get_buffer(ws));

        // the leader of each block records its block index in the global workspace
        if(thread_id == 0)
        {
          block_indices[block_id] = block_id;
        }

        // use the global barrier
        arrive_and_wait(get_barrier(ws));

        // the first thread of the first block checks that each block was recorded in the global workspace
        if(thread_id == 0 and block_id == 0)
        {
          int expected = 0;
          for(int id : block_indices)
          {
            assert(expected == id);
            ++expected;
          }
        }
      }

      // test the local workspace
      {
        std::span<int> thread_indices = reinterpret_buffer<int>(get_buffer(get_local_workspace(ws)));

        // each thread records its index in the local workspace
        thread_indices[thread_id] = thread_id;

        // use the local barrier
        arrive_and_wait(get_barrier(get_local_workspace(ws)));

        // the first thread of each block checks that each thread was recorded in the local workspace
        if(thread_id == 0)
        {
          int expected = 0;
          for(int id : thread_indices)
          {
            assert(expected == id);
            ++expected;
          }
        }
      }
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
void test_execute_kernel_customization_point(ns::cuda::coop_graph_executor ex, C shape)
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


void test(ns::cuda::coop_graph_executor ex)
{
  test_equality(ex);
  test_first_execute(ex);
  test_execute_after(ex);

  test_bulk_execute_after_member_function(ex);
  test_bulk_execute_after_customization_point(ex);

  test_bulk_execute_with_workspace_after_member_function(ex);
  test_bulk_execute_with_workspace_after_customization_point(ex);

  test_execute_kernel_customization_point(ex, array_shape[0]*array_shape[1]);
  test_execute_kernel_customization_point(ex, ns::int2{array_shape[0], array_shape[1]});
}


void test_with_stream(cudaStream_t s)
{
  cudaGraph_t g{};
  assert(cudaSuccess == cudaGraphCreate(&g,0));

  ns::cuda::coop_graph_executor ex{g, 0, s};

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


void test_coop_graph_executor()
{
  test_concepts();
  test_with_default_stream();
  test_with_new_stream();
}

