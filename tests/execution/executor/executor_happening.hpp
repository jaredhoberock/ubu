#include <ubu/execution/executor/executor_happening.hpp>

#undef NDEBUG
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

namespace ns = ubu;


std::future<void> make_ready_future()
{
  std::promise<void> p;
  auto result = p.get_future();
  p.set_value();
  return result;
}


struct upstream_executor_with_execute_member_function
{
  bool operator==(const upstream_executor_with_execute_member_function&) const { return true; }
  bool operator!=(const upstream_executor_with_execute_member_function&) const { return false; }

  std::future<void> execute(auto&& f) const
  {
    f();
    return make_ready_future();
  }
};


struct upstream_executor_with_execute_free_function
{
  bool operator==(const upstream_executor_with_execute_free_function&) const { return true; }
  bool operator!=(const upstream_executor_with_execute_free_function&) const { return false; }
};

std::future<void> execute(const upstream_executor_with_execute_free_function&, auto&& f)
{
  f();
  return make_ready_future();
}


void test()
{
  {
    static_assert(std::is_same_v<std::future<void>, ns::executor_happening_t<upstream_executor_with_execute_member_function>>, "Expected std::future<void>.");
  }

  {
    static_assert(std::is_same_v<std::future<void>, ns::executor_happening_t<upstream_executor_with_execute_free_function>>, "Expected std::future<void>.");
  }
}


void test_executor_happening()
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

