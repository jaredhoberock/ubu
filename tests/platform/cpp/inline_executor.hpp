#include <ubu/causality/initial_happening.hpp>
#include <ubu/execution/executor/concepts/executor.hpp>
#include <ubu/platform/cpp/inline_executor.hpp>

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

    auto before = ns::initial_happening(e);

    ns::execute_after(e, before, [&invoked]
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

