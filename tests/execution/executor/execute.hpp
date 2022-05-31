#include <ubu/execution/executor/execute.hpp>

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

struct has_execute_member
{
  template<class F>
  void execute(F&& f) const
  {
    f();
  }
};


struct has_execute_free_function {};

template<class F>
void execute(const has_execute_free_function&, F&& f)
{
  f();
}


void test()
{
  {
    has_execute_member e;

    bool invoked = false;
    ns::execute(e, [&]{ invoked = true; });
    assert(invoked);
  }

  {
    has_execute_free_function e;

    bool invoked = false;
    ns::execute(e, [&]{ invoked = true; });
    assert(invoked);
  }
}

void test_execute()
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

