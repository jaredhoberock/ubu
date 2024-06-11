#include <ubu/places/causality/wait.hpp>
#include <ubu/places/execution/executors/concepts/executor.hpp>
#include <ubu/places/execution/executors/finally_execute_after.hpp>
#include <ubu/places/execution/executors/first_execute.hpp>
#include <ubu/platforms/cuda/callback_executor.hpp>

#undef NDEBUG
#include <cassert>

#include <thread>


namespace ns = ubu;

void test(cudaStream_t s)
{
  {
    auto lambda = []{};
    static_assert(ns::executor_of<ns::cuda::callback_executor, decltype(lambda)>);
  }

  ns::cuda::callback_executor ex1{s};

  assert(s == ex1.stream());
  
  {
    int result = 0;
    int expected = 13;

    auto e = ns::first_execute(ex1, [&result,expected]
    {
      result = expected;
    });

    ns::wait(e);
    assert(expected == result);
  }

  {
    int result1 = 0;
    int expected1 = 13;
    
    auto e1 = ns::first_execute(ex1, [&result1,expected1]
    {
      result1 = expected1;
    });

    int result2 = 0;
    int expected2 = 7;
    auto e2 = ex1.execute_after(e1, [&result1,expected1,&result2,expected2]
    {
      assert(expected1 == result1);
      result2 = expected2;
    });

    ns::wait(e2);
    assert(expected2 == result2);
  }

  {
    int result1 = 0;
    int expected1 = 13;
    
    auto e1 = ns::first_execute(ex1, [&result1,expected1]
    {
      result1 = expected1;
    });

    int result2 = 0;
    int expected2 = 7;
    auto e2 = ex1.execute_after(e1, [&result1,expected1,&result2,expected2]
    {
      assert(expected1 == result1);
      result2 = expected2;
    });

    int result3 = 0;
    int expected3 = 42;
    ns::finally_execute_after(ex1, e2, [&result2,expected2,&result3,expected3]
    {
      assert(expected2 == result2);
      result3 = expected3;
    });

    assert(cudaStreamSynchronize(s) == cudaSuccess);
    assert(expected3 == result3);
  }

  ns::cuda::callback_executor ex2{s};

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));
}


void test_on_default_stream()
{
  test(cudaStream_t{});
}


void test_on_new_stream()
{
  cudaStream_t s{};
  assert(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);

  test(s);

  assert(cudaStreamDestroy(s) == cudaSuccess);
}


void test_callback_executor()
{
  test_on_default_stream();
  test_on_new_stream();
}

