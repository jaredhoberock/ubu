#include <ubu/causality/actual_happening.hpp>

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


struct actual_happening_with_member_functions
{
  static actual_happening_with_member_functions initial_happening()
  {
    return {};
  }

  bool has_happened() const
  {
    return true;
  }
};


struct actual_happening_with_has_happened_member
{
  bool has_happened() const
  {
    return true;
  }
};

actual_happening_with_has_happened_member initial_happening(actual_happening_with_has_happened_member)
{
  return {};
}


struct actual_happening_with_free_functions {};

actual_happening_with_free_functions initial_happening(actual_happening_with_free_functions)
{
  return {};
}

bool has_happened(const actual_happening_with_free_functions&)
{
  return true;
}


struct actual_happening_with_has_happened_free_function
{
  static actual_happening_with_has_happened_free_function initial_happening()
  {
    return {};
  }
};

bool has_happened(const actual_happening_with_has_happened_free_function&)
{
  return true;
}


void test()
{
  static_assert(ns::actual_happening<actual_happening_with_member_functions>);

  static_assert(ns::actual_happening<actual_happening_with_has_happened_member>);

  static_assert(ns::actual_happening<actual_happening_with_free_functions>);

  static_assert(ns::actual_happening<actual_happening_with_has_happened_free_function>);

  static_assert(ns::actual_happening<std::future<void>>);
}


void test_actual_happening()
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

