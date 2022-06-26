#include <ubu/causality/wait.hpp>
#include <ubu/cuda/callback_executor.hpp>
#include <ubu/execution/executor/executor.hpp>

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
  //assert(ns::blocking.possibly == ex1.query(ns::blocking));
  
  {
    int result = 0;
    int expected = 13;

    ex1.execute([&result,expected]
    {
      result = expected;
    });

    assert(cudaStreamSynchronize(s) == cudaSuccess);
    assert(expected == result);
  }

  {
    int result = 0;
    int expected = 13;

    auto e = ex1.first_execute([&result,expected]
    {
      result = expected;
    });

    ns::wait(e);
    assert(expected == result);
  }

  {
    int result1 = 0;
    int expected1 = 13;
    
    auto e1 = ex1.first_execute([&result1,expected1]
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
    
    auto e1 = ex1.first_execute([&result1,expected1]
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
    ex1.finally_execute(e2, [&result2,expected2,&result3,expected3]
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

