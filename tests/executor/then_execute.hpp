#include <aspera/event/complete_event.hpp>
#include <aspera/executor/inline_executor.hpp>
#include <aspera/executor/then_execute.hpp>

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

namespace ns = aspera;

struct has_then_execute_member_function
{
  template<ns::event E, class F>
  ns::complete_event then_execute(E&& before, F&& f) const
  {
    ns::wait(std::move(before));
    f();
    return {};
  }
};


struct has_then_execute_free_function {};

template<ns::event E, class F>
ns::complete_event then_execute(const has_then_execute_free_function&, E&& before, F&& f)
{
  ns::wait(std::move(before));
  f();
  return {};
}


void test()
{
  {
    has_then_execute_member_function ex;
    ns::complete_event before;

    bool invoked = false;
    auto e = ns::then_execute(ex, before, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }

  {
    has_then_execute_free_function ex;
    ns::complete_event before;

    bool invoked = false;
    auto e = ns::then_execute(ex, before, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }

  {
    ns::inline_executor ex;
    ns::complete_event before;

    bool invoked = false;
    auto e = ns::then_execute(ex, before, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }
}

void test_then_execute()
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

