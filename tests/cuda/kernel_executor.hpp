#include <aspera/coordinate/colexicographic_index.hpp>
#include <aspera/coordinate/colexicographic_index_to_grid_coordinate.hpp>
#include <aspera/coordinate/grid_coordinate.hpp>
#include <aspera/coordinate/lattice.hpp>
#include <aspera/cuda/kernel_executor.hpp>
#include <aspera/event/make_independent_event.hpp>
#include <aspera/event/wait.hpp>
#include <aspera/execution/executor/bulk_execute_after.hpp>
#include <aspera/execution/executor/bulk_execution_grid.hpp>
#include <aspera/execution/executor/execute.hpp>
#include <aspera/execution/executor/execute_after.hpp>
#include <aspera/execution/executor/executor.hpp>
#include <aspera/execution/executor/finally_execute_after.hpp>
#include <aspera/execution/executor/first_execute.hpp>

#undef NDEBUG
#include <cassert>

#include <cuda_runtime_api.h>


namespace ns = aspera;

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __managed__
#define __managed__
#endif

#ifndef __global__
#define __global__
#endif


__managed__ int result;
__managed__ int result1;
__managed__ int result2;


void test_equality(cudaStream_t s, std::size_t dynamic_shared_memory_size, int d)
{
  using namespace ns;

  cuda::kernel_executor ex1{d, s};
  
  cuda::kernel_executor ex2 = ex1;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


void test_execute(cudaStream_t s, int d)
{
  using namespace ns;

  cuda::kernel_executor ex1{d, s};

  result = 0;
  int expected = 13;

  try
  {
    ns::execute(ex1,[=] 
    {
      result = expected;
    });

    assert(cudaStreamSynchronize(s) == cudaSuccess);
    assert(expected == result);
  }
  catch(std::runtime_error&)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


void test_first_execute(cudaStream_t s, int d)
{
  using namespace ns;

  cuda::kernel_executor ex1{d, s};

  result = 0;
  int expected = 13;

  try
  {
    auto e = ns::first_execute(ex1, [expected]
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


void test_execute_after(cudaStream_t s, int d)
{
  using namespace ns;

  cuda::kernel_executor ex{d, s};

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


void test_finally_execute_after(cudaStream_t s, int d)
{
  using namespace ns;

  cuda::kernel_executor ex{d, s};

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

    assert(cudaStreamSynchronize(s) == cudaSuccess);
    assert(expected2 == result2);
  }
  catch(std::runtime_error&)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


int hash_coord(ns::cuda::thread_id coord)
{
  return coord.block.x ^ coord.block.y ^ coord.block.z ^ coord.thread.x ^ coord.thread.y ^ coord.thread.z;
}


// this array has blockIdx X threadIdx axes
// put 4 elements in each axis
__managed__ int bulk_result[4][4][4][4][4][4] = {};


void test_native_bulk_execute_after(cudaStream_t s, int d)
{
  using namespace ns;

  cuda::kernel_executor ex1{d, s};

  cuda::kernel_executor::coordinate_type shape{{4,4,4}, {4,4,4}};

  try
  {
    cuda::event before = ns::make_independent_event(ex1);

    cuda::event e = ns::bulk_execute_after(ex1, before, shape, [=](ns::cuda::thread_id coord)
    {
      int result = hash_coord(coord);

      bulk_result[coord.block.x][coord.block.y][coord.block.z][coord.thread.x][coord.thread.y][coord.thread.z] = result;
    });

    ns::wait(e);

    // XXX it would be much nicer to just iterate through a lattice
    for(int bx = 0; bx != shape.block.x; ++bx)
    {
      for(int by = 0; by != shape.block.y; ++by)
      {
        for(int bz = 0; bz != shape.block.z; ++bz)
        {
          for(int tx = 0; tx != shape.thread.x; ++tx)
          {
            for(int ty = 0; ty != shape.thread.y; ++ty)
            {
              for(int tz = 0; tz != shape.thread.z; ++tz)
              {
                cuda::thread_id coord{{bx,by,bz}, {tx,ty,tz}};
                unsigned int expected = hash_coord(coord);

                assert(expected == bulk_result[coord.block.x][coord.block.y][coord.block.z][coord.thread.x][coord.thread.y][coord.thread.z]);
              }
            }
          }
        }
      }
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


template<ns::grid_coordinate C>
void test_ND_bulk_execute_after(cudaStream_t s, int d, C shape)
{
  using namespace ns;

  cuda::kernel_executor ex1{d, s};

  try
  {
    cuda::event before = ns::make_independent_event(ex1);

    ns::int6 bulk_result_shape{4,4,4,4,4,4};

    cuda::event e = ns::bulk_execute_after(ex1, before, shape, [=](C coord)
    {
      int i = colexicographic_index(coord, shape);
      int6 a = colexicographic_index_to_grid_coordinate(i, bulk_result_shape);

      bulk_result[a[0]][a[1]][a[2]][a[3]][a[4]][a[5]] = i;
    });

    ns::wait(e);

    for(auto a : ns::lattice{bulk_result_shape})
    {
      int coord = colexicographic_index(a, bulk_result_shape);

      assert(coord == bulk_result[a[0]][a[1]][a[2]][a[3]][a[4]][a[5]]);
    }
  }
  catch(std::runtime_error& e)
  {
#if defined(__CUDACC__)
    assert(false);
#endif
  }
}


void test_on_stream(cudaStream_t s)
{
  test_equality(s, 16, 0);
  test_execute(s, 0);
  test_finally_execute_after(s, 0);
  test_first_execute(s, 0);
  test_execute_after(s, 0);
  test_native_bulk_execute_after(s, 0);

  test_ND_bulk_execute_after(s, 0, 4*4*4*4*4*4);
  test_ND_bulk_execute_after(s, 0, ns::int2{4*4*4, 4*4*4});
  test_ND_bulk_execute_after(s, 0, ns::int3{4*4, 4*4, 4*4});
  test_ND_bulk_execute_after(s, 0, ns::int4{4*4, 4*4, 4, 4});
  test_ND_bulk_execute_after(s, 0, ns::int5{4*4, 4, 4, 4, 4});
}


void test_on_default_stream()
{
  cudaStream_t s{};
  test_on_stream(s);
}


void test_on_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  test_on_stream(s);

  assert(cudaStreamDestroy(s) == cudaSuccess);
}


void test_concepts()
{
  static_assert(ns::grid_coordinate<ns::cuda::kernel_executor::coordinate_type>);
  static_assert(std::same_as<ns::cuda::kernel_executor::coordinate_type, ns::executor_coordinate_t<ns::cuda::kernel_executor>>);
}


void test_bulk_execution_grid()
{
  using namespace ns;

  cuda::kernel_executor ex;
  cuda::thread_id result = bulk_execution_grid(ex, 128);

  assert(1 == result.block.x);
  assert(1 == result.block.y);
  assert(1 == result.block.z);

  assert(128 == result.thread.x);
  assert(  1 == result.thread.y);
  assert(  1 == result.thread.z);
}


void test_kernel_executor()
{
  test_concepts();
  static_assert(ns::executor<ns::cuda::kernel_executor>);
  test_bulk_execution_grid();
  test_on_default_stream();
  test_on_new_stream();
}

