#include <aspera/event/complete_event.hpp>
#include <aspera/executor/finally_execute.hpp>
#include <aspera/executor/inline_executor.hpp>
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

struct has_finally_execute_member_function
{
  template<ns::event E, class F>
  void finally_execute(E&& before, F&& f) const
  {
    ns::wait(std::move(before));
    f();
  }
};


struct has_finally_execute_free_function {};

template<ns::event E, class F>
void finally_execute(const has_finally_execute_free_function&, E&& before, F&& f)
{
  ns::wait(std::move(before));
  f();
}


void test()
{
  {
    has_finally_execute_member_function ex;
    ns::complete_event before;

    bool invoked = false;
    ns::finally_execute(ex, before, [&]{ invoked = true; });
    assert(invoked);
  }

  {
    has_finally_execute_free_function ex;
    ns::complete_event before;

    bool invoked = false;
    ns::finally_execute(ex, before, [&]{ invoked = true; });
    assert(invoked);
  }

  {
    ns::inline_executor ex;
    ns::complete_event before;

    bool invoked = false;
    ns::finally_execute(ex, before, [&]{ invoked = true; });
    assert(invoked);
  }
}

void test_finally_execute()
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

