#include <ubu/places/execution/executor/execute_kernel.hpp>
#include <ubu/platform/cpp/inline_executor.hpp>

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

struct has_execute_kernel_member
{
  template<class F>
  void execute_kernel(int n, F&& f) const
  {
    for(int i = 0; i < n; ++i)
    {
      f(i);
    }
  }
};


struct has_execute_kernel_free_function {};

template<class F>
void execute_kernel(const has_execute_kernel_free_function&, int n, F&& f)
{
  for(int i = 0; i < n; ++i)
  {
    f(i);
  }
}


void test()
{
  {
    has_execute_kernel_member e;

    int counter = 0;
    int expected = 10;

    ns::execute_kernel(e, expected, [&](int){ ++counter; });
    assert(expected == counter);
  }

  {
    has_execute_kernel_free_function e;

    int counter = 0;
    int expected = 10;

    ns::execute_kernel(e, expected, [&](int){ ++counter; });
    assert(expected == counter);
  }

  {
    // 1D shape with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;
    int expected = 10;

    ns::execute_kernel(e, expected, [&](int){ ++counter; });
    assert(expected == counter);
  }

  {
    // 2D shape with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;
    ns::int2 shape{2,5};

    ns::execute_kernel(e, shape, [&](ns::int2){ ++counter; });
    assert(shape.product() == counter);
  }

  {
    // 3D shape with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;
    ns::int3 shape{2,5,7};

    ns::execute_kernel(e, shape, [&](ns::int3){ ++counter; });
    assert(shape.product() == counter);
  }
}

void test_execute_kernel()
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

