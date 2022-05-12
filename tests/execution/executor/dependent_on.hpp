#include <aspera/event/always_complete_event.hpp>
#include <aspera/event/wait.hpp>
#include <aspera/execution/executor/dependent_on.hpp>
#include <aspera/execution/executor/first_execute.hpp>
#include <aspera/execution/executor/inline_executor.hpp>

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


struct has_dependent_on_member_function
{
  template<class... Events>
  ns::always_complete_event dependent_on(Events&&... events) const
  {
    swallow(call_wait(events)...);

    return {};
  }
};


void test()
{
  {
    has_dependent_on_member_function ex;
    auto e = ns::dependent_on(ex, ns::always_complete_event{}, ns::always_complete_event{}, ns::always_complete_event{});
    ns::wait(e);
  }


  {
    ns::inline_executor ex;

    int expected = 3;
    int counter = 0;

    ns::always_complete_event e1 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    ns::always_complete_event e2 = ns::first_execute(ex, [&counter]
    {
      ++counter;
    });

    ns::always_complete_event e3 = ns::first_execute(ex, [&counter]
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

