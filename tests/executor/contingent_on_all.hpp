#include <aspera/event/wait.hpp>
#include <aspera/executor/contingent_on_all.hpp>
#include <aspera/executor/first_execute.hpp>
#include <aspera/executor/inline_executor.hpp>
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


struct has_contingent_on_all_member_function
{
  template<class C>
  ns::complete_event contingent_on_all(C&& events) const
  {
    for(auto& e : events)
    {
      ns::wait(e);
    }

    return {};
  }
};


struct has_contingent_on_all_free_function {};

template<class C>
ns::complete_event contingent_on_all(const has_contingent_on_all_free_function &, C&& events)
{
  for(auto& e : events)
  {
    ns::wait(e);
  }

  return {};
}


void test()
{
  {
    has_contingent_on_all_member_function ex;
    std::vector<ns::complete_event> events(10);

    auto e = ns::contingent_on_all(ex, std::move(events));
    ns::wait(e);
  }


  {
    has_contingent_on_all_free_function ex;
    std::vector<ns::complete_event> events(10);

    auto e = ns::contingent_on_all(ex, std::move(events));
    ns::wait(e);
  }

  {
    ns::inline_executor ex;

    std::vector<ns::complete_event> events;
    int expected = 10;
    int counter = 0;
    for(auto i = 0; i < expected; ++i)
    {
      events.push_back(ns::first_execute(ex, [&counter]
      {
        ++counter;
      }));
    }

    auto e = ns::contingent_on_all(ex, std::move(events));
    ns::wait(e);
    assert(expected == counter);
  }
}


void test_contingent_on_all()
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

