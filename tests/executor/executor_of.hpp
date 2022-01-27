#include <aspera/executor/executor_of.hpp>

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


struct executor_with_execute_member
{
  bool operator==(const executor_with_execute_member&) const { return true; }
  bool operator!=(const executor_with_execute_member&) const { return false; }

  template<class F>
  void execute(F&& f) const
  {
    f();
  }
};


struct executor_with_execute_free_function
{
  bool operator==(const executor_with_execute_free_function&) const { return true; }
  bool operator!=(const executor_with_execute_free_function&) const { return false; }
};

template<class F>
void execute(const executor_with_execute_free_function&, F&& f)
{
  f();
}


void test()
{
  {
    auto lambda = []{};

    static_assert(ns::executor_of<executor_with_execute_member, decltype(lambda)>);
  }

  {
    auto lambda = []{};

    static_assert(ns::executor_of<executor_with_execute_free_function, decltype(lambda)>);
  }
}


void test_executor_of()
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

