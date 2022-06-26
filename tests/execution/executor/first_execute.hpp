#include <ubu/event/past_event.hpp>
#include <ubu/execution/executor/first_execute.hpp>
#include <ubu/execution/executor/inline_executor.hpp>

#undef NDEBUG
#include <cassert>

#include <thread>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = ubu;

struct has_first_execute_member_function
{
  template<class F>
  ns::past_event first_execute(F&& f) const
  {
    f();
    return {};
  }
};


struct has_first_execute_free_function {};

template<class F>
ns::past_event first_execute(const has_first_execute_free_function&, F&& f)
{
  f();
  return {};
}


void test()
{
  {
    has_first_execute_member_function ex;

    bool invoked = false;
    auto e = ns::first_execute(ex, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }

  {
    has_first_execute_free_function ex;

    bool invoked = false;
    auto e = ns::first_execute(ex, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }

  {
    ns::inline_executor ex;

    bool invoked = false;
    auto e = ns::first_execute(ex, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }
}

void test_first_execute()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

