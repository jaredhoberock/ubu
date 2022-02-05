#include <aspera/event/always_complete_event.hpp>
#include <aspera/executor/finally_execute.hpp>

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
    auto lambda = []{};

    static_assert(std::is_same_v<void, ns::finally_execute_result_t<has_finally_execute_member_function, ns::always_complete_event, decltype(lambda)>>, "Expected void.");
  }

  {
    auto lambda = []{};

    static_assert(std::is_same_v<void, ns::finally_execute_result_t<has_finally_execute_free_function, ns::always_complete_event, decltype(lambda)>>, "Expected void.");
  }
}

void test_finally_execute_result_t()
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

