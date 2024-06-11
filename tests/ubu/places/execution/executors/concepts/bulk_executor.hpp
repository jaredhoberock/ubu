#include <ubu/places/causality/past_event.hpp>
#include <ubu/places/execution/executors/concepts/bulk_executor.hpp>
#include <ubu/platforms/cpp/inline_executor.hpp>

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


struct has_bulk_execute_after_member : public ns::cpp::inline_executor
{
  ns::past_event bulk_execute_after(ns::past_event, int n, int workspace_shape, auto&& f) const
  {
    std::vector<std::byte> buffer(workspace_shape);
    std::span workspace(buffer.data(), buffer.size());

    for(int i = 0; i != n; ++i)
    {
      f(i, workspace);
    }

    return {};
  }
};


struct has_bulk_execute_after_free_function : public ns::cpp::inline_executor
{
};

ns::past_event bulk_execute_after(const has_bulk_execute_after_free_function&, ns::past_event, int n, int workspace_shape, auto&& f)
{
  std::vector<std::byte> buffer(workspace_shape);
  std::span workspace(buffer.data(), buffer.size());

  for(int i = 0; i != n; ++i)
  {
    f(i, workspace);
  }

  return {};
}


void test()
{
  {
    static_assert(ns::bulk_executor<has_bulk_execute_after_member>);
  }

  {
    static_assert(ns::executor<has_bulk_execute_after_free_function>);
  }
}


void test_bulk_executor()
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

