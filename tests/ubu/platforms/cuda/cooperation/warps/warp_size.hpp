#include <ubu/platforms/cuda/cooperators/warps.hpp>

void test_warp_size()
{
  static_assert(32 == ubu::cuda::warp_size);
}

