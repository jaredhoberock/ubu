#include <aspera/executor/event_executor.hpp>
#include <cassert>
#include <future>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = aspera;


std::future<void> make_ready_future()
{
  std::promise<void> p;
  auto result = p.get_future();
  p.set_value();
  return result;
}


struct event_executor_with_execute_member
{
  bool operator==(const event_executor_with_execute_member&) const { return true; }
  bool operator!=(const event_executor_with_execute_member&) const { return false; }

  std::future<void> execute(auto&& f) const
  {
    f();
    return make_ready_future();
  }
};


struct event_executor_with_execute_free_function
{
  bool operator==(const event_executor_with_execute_free_function&) const { return true; }
  bool operator!=(const event_executor_with_execute_free_function&) const { return false; }
};

std::future<void> execute(const event_executor_with_execute_free_function&, auto&& f)
{
  f();
  return make_ready_future();
}


void test()
{
  {
    static_assert(ns::event_executor<event_executor_with_execute_member>);
  }

  {
    static_assert(ns::event_executor<event_executor_with_execute_free_function>);
  }
}


void test_event_executor()
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

