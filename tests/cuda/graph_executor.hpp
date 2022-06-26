#include <ubu/causality/first_cause.hpp>
#include <ubu/causality/wait.hpp>
#include <ubu/coordinate/colexicographic_index.hpp>
#include <ubu/coordinate/colexicographic_index_to_grid_coordinate.hpp>
#include <ubu/coordinate/grid_coordinate.hpp>
#include <ubu/coordinate/lattice.hpp>
#include <ubu/cuda/graph_executor.hpp>
#include <ubu/execution/executor/bulk_execute_after.hpp>
#include <ubu/execution/executor/bulk_execution_grid.hpp>
#include <ubu/execution/executor/execute.hpp>
#include <ubu/execution/executor/execute_after.hpp>
#include <ubu/execution/executor/executor.hpp>
#include <ubu/execution/executor/finally_execute_after.hpp>
#include <ubu/execution/executor/first_execute.hpp>

#undef NDEBUG
#include <cassert>

#include <cuda_runtime_api.h>


namespace ns = ubu;

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


void test_equality(ns::cuda::graph_executor ex1)
{
  using namespace ns;
  
  cuda::graph_executor ex2 = ex1;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


// XXX not clear how to test execute with graph_executor
//void test_execute(cudaStream_t s, int d)
//{
//  using namespace ns;
//
//  cuda::kernel_executor ex1{d, s};
//
//  result = 0;
//  int expected = 13;
//
//  try
//  {
//    ns::execute(ex1,[=] 
//    {
//      result = expected;
//    });
//
//    assert(cudaStreamSynchronize(s) == cudaSuccess);
//    assert(expected == result);
//  }
//  catch(std::runtime_error&)
//  {
//#if defined(__CUDACC__)
//    assert(false);
//#endif
//  }
//}


void test_first_execute(ns::cuda::graph_executor ex)
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


void test_execute_after(ns::cuda::graph_executor ex)
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


// XXX not clear how to test finally_ for graph_executor
//void test_finally_execute_after(cudaStream_t s, int d)
//{
//  using namespace ns;
//
//  cuda::kernel_executor ex{d, s};
//
//  result1 = 0;
//  int expected1 = 13;
//  
//  try
//  {
//    auto e1 = ns::first_execute(ex, [expected1]
//    {
//      result1 = expected1;
//    });
//
//    result2 = 0;
//    int expected2 = 7;
//    ns::finally_execute_after(ex, e1, [expected1,expected2]
//    {
//      assert(expected1 == result1);
//      result2 = expected2;
//    });
//
//    assert(cudaStreamSynchronize(s) == cudaSuccess);
//    assert(expected2 == result2);
//  }
//  catch(std::runtime_error&)
//  {
//#if defined(__CUDACC__)
//    assert(false);
//#endif
//  }
//}


int hash_coord(ns::cuda::thread_id coord)
{
  return coord.block.x ^ coord.block.y ^ coord.block.z ^ coord.thread.x ^ coord.thread.y ^ coord.thread.z;
}


// this array has blockIdx X threadIdx axes
// put 4 elements in each axis
__managed__ int bulk_result[4][4][4][4][4][4] = {};


void test_native_bulk_execute_after(ns::cuda::graph_executor ex)
{
  using namespace ns;

  cuda::kernel_executor::coordinate_type shape{{4,4,4}, {4,4,4}};

  try
  {
    auto before = ns::first_cause(ex);

    auto e = ns::bulk_execute_after(ex, before, shape, [=](ns::cuda::thread_id coord)
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
void test_ND_bulk_execute_after(ns::cuda::graph_executor ex, C shape)
{
  using namespace ns;

  try
  {
    auto before = ns::first_cause(ex);

    ns::int6 bulk_result_shape{4,4,4,4,4,4};

    auto e = ns::bulk_execute_after(ex, before, shape, [=](C coord)
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


void test(ns::cuda::graph_executor ex)
{
  test_equality(ex);
  //test_execute(ex);
  test_first_execute(ex);
  test_execute_after(ex);
  //test_finally_execute_after(ex);
  test_native_bulk_execute_after(ex);

  test_ND_bulk_execute_after(ex, 4*4*4*4*4*4);
  test_ND_bulk_execute_after(ex, ns::int2{4*4*4, 4*4*4});
  test_ND_bulk_execute_after(ex, ns::int3{4*4, 4*4, 4*4});
  test_ND_bulk_execute_after(ex, ns::int4{4*4, 4*4, 4, 4});
  test_ND_bulk_execute_after(ex, ns::int5{4*4, 4, 4, 4, 4});
}


void test_on_default_stream()
{
  cudaGraph_t g{};
  assert(cudaSuccess == cudaGraphCreate(&g,0));

  ns::cuda::graph_executor ex{g, 0, 0};

  test(ex);

  assert(cudaSuccess == cudaGraphDestroy(g));
}


void test_on_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  cudaGraph_t g{};
  assert(cudaSuccess == cudaGraphCreate(&g,0));

  ns::cuda::graph_executor ex{g, 0, s};

  test(ex);

  assert(cudaSuccess == cudaGraphDestroy(g));
  assert(cudaStreamDestroy(s) == cudaSuccess);
}


void test_concepts()
{
  static_assert(ns::grid_coordinate<ns::cuda::graph_executor::coordinate_type>);
  static_assert(std::same_as<ns::cuda::graph_executor::coordinate_type, ns::executor_coordinate_t<ns::cuda::graph_executor>>);
  static_assert(ns::executor<ns::cuda::graph_executor>);
}


void test_bulk_execution_grid()
{
  using namespace ns;

  cudaGraph_t g{};
  assert(cudaSuccess == cudaGraphCreate(&g,0));

  cuda::graph_executor ex{g};
  cuda::thread_id result = bulk_execution_grid(ex, 128);

  assert(1 == result.block.x);
  assert(1 == result.block.y);
  assert(1 == result.block.z);

  assert(128 == result.thread.x);
  assert(  1 == result.thread.y);
  assert(  1 == result.thread.z);

  assert(cudaSuccess == cudaGraphDestroy(g));
}


void test_graph_executor()
{
  test_concepts();
  test_bulk_execution_grid();
  test_on_default_stream();
//  test_on_new_stream();
}

