#include <aspera/event/wait.hpp>

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

struct has_wait_member
{
  int wait() const
  {
    return 13;
  }
};


struct has_wait_free_function {};

int wait(const has_wait_free_function&)
{
  return 13;
}


void test()
{
  {
    has_wait_member e;

    auto result = ns::wait(e);
    assert(13 == result);
  }

  {
    has_wait_free_function e;

    auto result = ns::wait(e);
    assert(13 == result);
  }
}

void test_wait()
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

