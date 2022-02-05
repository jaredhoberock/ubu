#include <aspera/event/always_complete_event.hpp>
#include <aspera/executor/finally_execute_after.hpp>
#include <aspera/executor/inline_executor.hpp>

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

struct has_finally_execute_after_member_function
{
  template<ns::event E, class F>
  void finally_execute_after(E&& before, F&& f) const
  {
    ns::wait(std::move(before));
    f();
  }
};


struct has_finally_execute_after_free_function {};

template<ns::event E, class F>
void finally_execute_after(const has_finally_execute_after_free_function&, E&& before, F&& f)
{
  ns::wait(std::move(before));
  f();
}


void test()
{
  {
    has_finally_execute_after_member_function ex;
    ns::always_complete_event before;

    bool invoked = false;
    ns::finally_execute_after(ex, before, [&]{ invoked = true; });
    assert(invoked);
  }

  {
    has_finally_execute_after_free_function ex;
    ns::always_complete_event before;

    bool invoked = false;
    ns::finally_execute_after(ex, before, [&]{ invoked = true; });
    assert(invoked);
  }

  {
    ns::inline_executor ex;
    ns::always_complete_event before;

    bool invoked = false;
    ns::finally_execute_after(ex, before, [&]{ invoked = true; });
    assert(invoked);
  }
}

void test_finally_execute_after()
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

