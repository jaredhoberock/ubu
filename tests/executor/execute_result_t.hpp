#include <aspera/executor/execute.hpp>

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

struct has_execute_member
{
  template<class F>
  int execute(F&& f) const
  {
    f();
    return 13;
  }
};


struct has_execute_free_function {};

template<class F>
double execute(const has_execute_free_function&, F&& f)
{
  f();
  return 13.;
}


void test()
{
  {
    auto lambda = []{};

    static_assert(std::is_same_v<int, ns::execute_result_t<has_execute_member, decltype(lambda)>>, "Expected int.");
  }

  {
    auto lambda = []{};

    static_assert(std::is_same_v<double, ns::execute_result_t<has_execute_free_function, decltype(lambda)>>, "Expected double.");
  }
}

void test_execute_result_t()
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


