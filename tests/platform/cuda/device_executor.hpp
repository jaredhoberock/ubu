#include <array>
#include <ubu/causality.hpp>
#include <ubu/execution/executor/bulk_execute_after.hpp>
#include <ubu/execution/executor/bulk_execution_grid.hpp>
#include <ubu/execution/executor/executor.hpp>
#include <ubu/execution/executor/finally_execute_after.hpp>
#include <ubu/execution/executor/first_execute.hpp>
#include <ubu/grid/coordinate/lift_coordinate.hpp>
#include <ubu/grid/layout/stride/apply_stride.hpp>
#include <ubu/grid/layout/stride/compact_column_major_stride.hpp>
#include <ubu/platform/cuda/device_executor.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


void test_bulk_execution_grid()
{
  using namespace ns;

  cuda::device_executor ex;
  cuda::thread_id result = bulk_execution_grid(ex, 128);

  assert(128 == result.thread.x);
  assert(  1 == result.thread.y);
  assert(  1 == result.thread.z);

  assert(1 == result.block.x);
  assert(1 == result.block.y);
  assert(1 == result.block.z);
}


void test_concepts()
{
  static_assert(ns::coordinate<ns::cuda::device_executor::coordinate_type>);
  static_assert(std::same_as<ns::cuda::device_executor::coordinate_type, ns::executor_coordinate_t<ns::cuda::device_executor>>);
  static_assert(ns::executor<ns::cuda::device_executor>);
}


void test_equality(ns::cuda::device_executor ex1)
{
  using namespace ns;
  
  cuda::device_executor ex2 = ex1;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


#ifndef __managed__
#define __managed__
#endif


__managed__ int result;
__managed__ int result1;
__managed__ int result2;


void test_first_execute(ns::cuda::device_executor ex)
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
    assert(false);
#endif
  }
}


void test_execute_after(ns::cuda::device_executor ex)
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
    assert(false);
#endif
  }
}


void test_finally_execute_after(ns::cuda::device_executor ex)
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
    assert(false);
#endif
  }
}


int hash(ns::cuda::thread_id coord)
{
  return coord.thread.x ^ coord.thread.y ^ coord.thread.z ^ coord.block.x ^ coord.block.y ^ coord.block.z;
}


// this array has 3 + 3 axes to match threadIdx + blockIdx
constexpr std::array<int,6> array_shape = {2, 4, 6, 8, 10, 12};
__managed__ int array[2][4][6][8][10][12] = {};

void test_bulk_execute_after_member_function(ns::cuda::device_executor ex)
{
  using namespace ns;

  // partition the array shape into the device_executor's shape type
  // such that nearby threads touch nearby addresses

  cuda::device_executor::coordinate_type shape
  {
    // (thread.x, thread.y, thread.z)
    {array_shape[5],array_shape[4],array_shape[3]},
    // (block.x, block.y, block.z)
    {array_shape[2],array_shape[1],array_shape[0]}
  };

  try
  {
    cuda::event before = ns::first_cause(ex);

    cuda::event e = ex.bulk_execute_after(before, shape, [=](ns::cuda::thread_id coord)
    {
      int result = hash(coord);

      array[coord.block.z][coord.block.y][coord.block.x][coord.thread.z][coord.thread.y][coord.thread.x] = result;
    });

    ns::wait(e);

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
    assert(false);
#endif
  }
}


template<ns::coordinate C>
void test_bulk_execute_after_customization_point(ns::cuda::device_executor ex, C shape)
{
  using namespace ns;

  try
  {
    cuda::event before = ns::first_cause(ex);

    cuda::event e = ns::bulk_execute_after(ex, before, shape, [=](C coord)
    {
      int i = apply_stride(coord, compact_column_major_stride(shape));
      auto c = lift_coordinate(i, array_shape);

      array[c[0]][c[1]][c[2]][c[3]][c[4]][c[5]] = i;
    });

    ns::wait(e);

    for(int i = 0; i < ns::shape_size(array_shape); ++i)
    {
      auto c = lift_coordinate(i, array_shape);

      assert(i == array[c[0]][c[1]][c[2]][c[3]][c[4]][c[5]]);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


void test(ns::cuda::device_executor ex)
{
  test_equality(ex);
  test_finally_execute_after(ex);
  test_first_execute(ex);
  test_execute_after(ex);

  test_bulk_execute_after_member_function(ex);

  test_bulk_execute_after_customization_point(ex, array_shape[0]*array_shape[1]*array_shape[2]*array_shape[3]*array_shape[4]*array_shape[5]);
  test_bulk_execute_after_customization_point(ex, ns::int2{array_shape[0]*array_shape[1]*array_shape[2], array_shape[3]*array_shape[4]*array_shape[5]});
  test_bulk_execute_after_customization_point(ex, ns::int3{array_shape[0]*array_shape[1], array_shape[2]*array_shape[3], array_shape[4]*array_shape[5]});
  test_bulk_execute_after_customization_point(ex, ns::int4{array_shape[0]*array_shape[1], array_shape[2]*array_shape[3], array_shape[4], array_shape[5]});
  test_bulk_execute_after_customization_point(ex, ns::int5{array_shape[0]*array_shape[1], array_shape[2], array_shape[3], array_shape[4], array_shape[5]});
  test_bulk_execute_after_customization_point(ex, array_shape);
}


void test_with_default_stream()
{
  cudaStream_t s{};

  ns::cuda::device_executor ex{0, s};
  test(ex);
}


void test_with_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  ns::cuda::device_executor ex{0, s};
  test(ex);

  assert(cudaStreamDestroy(s) == cudaSuccess);
}


void test_device_executor()
{
  test_bulk_execution_grid();
  test_concepts();
  test_with_default_stream();
  test_with_new_stream();
}
