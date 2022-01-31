#include <aspera/event/wait.hpp>
#include <aspera/executor/contingent_on.hpp>
#include <aspera/executor/first_execute.hpp>
#include <aspera/executor/inline_executor.hpp>

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

namespace ns = aspera;


template<class... Args>
void swallow(Args&&...){};


template<class E>
int call_wait(E& e)
{
  ns::wait(e);
  return 0;
}


struct has_contingent_on_member_function
{
  template<class... Events>
  ns::complete_event contingent_on(Events&&... events) const
  {
    swallow(call_wait(events)...);

    return {};
  }
};


struct has_contingent_on_free_function {};

template<class... Events>
ns::complete_event contingent_on(const has_contingent_on_free_function &, Events&&... events)
{
  swallow(call_wait(events)...);

  return {};
}


void test()
{
  {
    has_contingent_on_member_function ex;
    auto e = ns::contingent_on(ex, ns::complete_event{}, ns::complete_event{}, ns::complete_event{});
    ns::wait(e);
  }


  {
    has_contingent_on_free_function ex;
    auto e = ns::contingent_on(ex, ns::complete_event{}, ns::complete_event{}, ns::complete_event{});
    ns::wait(e);
  }

  {
    ns::inline_executor ex;

    int expected = 3;
    int counter = 0;

    ns::complete_event e1 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    ns::complete_event e2 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    ns::complete_event e3 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    auto e = ns::contingent_on(ex, std::move(e1), std::move(e2), std::move(e3));
    ns::wait(e);

    assert(expected == counter);
  }
}


void test_contingent_on()
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


