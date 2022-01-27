#include <aspera/executor/executor.hpp>

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

  void execute(auto&& f) const
  {
    f();
  }
};


struct executor_with_execute_free_function
{
  bool operator==(const executor_with_execute_free_function&) const { return true; }
  bool operator!=(const executor_with_execute_free_function&) const { return false; }
};

void execute(const executor_with_execute_free_function&, auto&& f)
{
  f();
}


void test()
{
  {
    static_assert(ns::executor<executor_with_execute_member>);
  }

  {
    static_assert(ns::executor<executor_with_execute_free_function>);
  }
}


void test_executor()
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

