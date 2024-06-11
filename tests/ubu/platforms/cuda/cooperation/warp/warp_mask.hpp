#include <ubu/platforms/cuda/cooperation/warp.hpp>

void test_warp_mask()
{
  static_assert(0xFFFFFFFF == ubu::cuda::warp_mask);
}

