#include <ubu/execution/executor/executor.hpp>
#include <ubu/execution/executor/executor_of.hpp>
#include <ubu/execution/executor/inline_executor.hpp>

#undef NDEBUG
#include <cassert>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = ubu;


void test()
{
  {
    auto lambda = []{};
    static_assert(ns::executor_of<ns::inline_executor, decltype(lambda)>);
  }

  {
    bool invoked = false;

    ns::inline_executor e;

    ns::execute(e, [&invoked]
    {
      invoked = true;
    });

    assert(invoked);
  }
}


void test_inline_executor()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] ()
  {
    test();
  });
#endif
}

