#include <ubu/places/execution/executor/traits/executor_happening.hpp>

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


struct executor_with_execute_after_member_function
{
  bool operator==(const executor_with_execute_after_member_function&) const { return true; }
  bool operator!=(const executor_with_execute_after_member_function&) const { return false; }

  std::future<void> initial_happening() const;

  std::future<void> execute_after(const std::future<void>& before, auto&& f) const
  {
    before.wait();
    f();
    return make_ready_future();
  }
};


struct executor_with_execute_after_free_function
{
  bool operator==(const executor_with_execute_after_free_function&) const { return true; }
  bool operator!=(const executor_with_execute_after_free_function&) const { return false; }

  std::future<void> initial_happening() const;
};

std::future<void> execute_after(const executor_with_execute_after_free_function&, const std::future<void>& before, auto&& f)
{
  before.wait();
  f();
  return make_ready_future();
}


void test()
{
  {
    static_assert(std::is_same_v<std::future<void>, ns::executor_happening_t<executor_with_execute_after_member_function>>, "Expected std::future<void>.");
  }

  {
    static_assert(std::is_same_v<std::future<void>, ns::executor_happening_t<executor_with_execute_after_free_function>>, "Expected std::future<void>.");
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

