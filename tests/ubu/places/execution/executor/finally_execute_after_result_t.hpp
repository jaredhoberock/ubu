#include <ubu/places/causality/happening.hpp>
#include <ubu/places/causality/past_event.hpp>
#include <ubu/places/causality/wait.hpp>
#include <ubu/places/execution/executor/finally_execute_after.hpp>

#undef NDEBUG
#include <cassert>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = ubu;

struct has_finally_execute_after_member_function
{
  template<ns::happening H, class F>
  void finally_execute_after(H&& before, F&& f) const
  {
    ns::wait(std::move(before));
    f();
  }
};


struct has_finally_execute_after_free_function {};

template<ns::happening H, class F>
void finally_execute_after(const has_finally_execute_after_free_function&, H&& before, F&& f)
{
  ns::wait(std::move(before));
  f();
}


void test()
{
  {
    auto lambda = []{};

    static_assert(std::is_same_v<void, ns::finally_execute_after_result_t<has_finally_execute_after_member_function, ns::past_event, decltype(lambda)>>, "Expected void.");
  }

  {
    auto lambda = []{};

    static_assert(std::is_same_v<void, ns::finally_execute_after_result_t<has_finally_execute_after_free_function, ns::past_event, decltype(lambda)>>, "Expected void.");
  }
}

void test_finally_execute_after_result_t()
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

