#include <aspera/event/always_complete_event.hpp>
#include <aspera/execution/executor/execute_after.hpp>
#include <aspera/execution/executor/inline_executor.hpp>

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

struct has_execute_after_member_function
{
  template<ns::event E, class F>
  ns::always_complete_event execute_after(E&& before, F&& f) const
  {
    ns::wait(std::move(before));
    f();
    return {};
  }
};


struct has_execute_after_free_function {};

template<ns::event E, class F>
ns::always_complete_event execute_after(const has_execute_after_free_function&, E&& before, F&& f)
{
  ns::wait(std::move(before));
  f();
  return {};
}


void test()
{
  {
    has_execute_after_member_function ex;
    ns::always_complete_event before;

    bool invoked = false;
    auto e = ns::execute_after(ex, before, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }

  {
    has_execute_after_free_function ex;
    ns::always_complete_event before;

    bool invoked = false;
    auto e = ns::execute_after(ex, before, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }

  {
    ns::inline_executor ex;
    ns::always_complete_event before;

    bool invoked = false;
    auto e = ns::execute_after(ex, before, [&]{ invoked = true; });
    ns::wait(e);
    assert(invoked);
  }
}

void test_execute_after()
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

