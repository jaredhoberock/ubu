#include <cassert>
#include <aspera/executor/wait.hpp>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = aspera;

struct has_wait_member
{
  int wait() const
  {
    return 13;
  }
};


struct has_wait_free_function {};

double wait(const has_wait_free_function&)
{
  return 13.;
}


void test()
{
  {
    static_assert(std::is_same_v<int, ns::wait_result_t<has_wait_member>>, "Expected int.");
  }

  {
    static_assert(std::is_same_v<double, ns::wait_result_t<has_wait_free_function>>, "Expected double.");
  }
}

void test_wait_result_t()
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

