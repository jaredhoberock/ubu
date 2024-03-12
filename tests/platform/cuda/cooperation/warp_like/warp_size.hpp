#include <ubu/platform/cuda/cooperation/warp_like.hpp>

void test_warp_size()
{
  static_assert(32 == ubu::cuda::warp_size);
}

