#include <ubu/platform/cuda/cooperation/warp_like.hpp>

void test_warp_mask()
{
  static_assert(0xFFFFFFFF == ubu::cuda::warp_mask);
}

