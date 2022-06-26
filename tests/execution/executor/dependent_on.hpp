#include <ubu/causality/past_event.hpp>
#include <ubu/causality/wait.hpp>
#include <ubu/execution/executor/dependent_on.hpp>
#include <ubu/execution/executor/first_execute.hpp>
#include <ubu/execution/executor/inline_executor.hpp>

#undef NDEBUG
#include <cassert>

#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = ubu;


template<class... Args>
void swallow(Args&&...){};


template<class E>
int call_wait(E& e)
{
  ns::wait(e);
  return 0;
}


struct has_dependent_on_member_function
{
  template<class... Events>
  ns::past_event dependent_on(Events&&... events) const
  {
    swallow(call_wait(events)...);

    return {};
  }
};


void test()
{
  {
    has_dependent_on_member_function ex;
    auto e = ns::dependent_on(ex, ns::past_event{}, ns::past_event{}, ns::past_event{});
    ns::wait(e);
  }


  {
    ns::inline_executor ex;

    int expected = 3;
    int counter = 0;

    ns::past_event e1 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    ns::past_event e2 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    ns::past_event e3 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    auto e = ns::dependent_on(ex, std::move(e1), std::move(e2), std::move(e3));
    ns::wait(e);

    assert(expected == counter);
  }
}


void test_dependent_on()
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

