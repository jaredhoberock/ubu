#include <ubu/coordinate/grid_coordinate.hpp>
#include <ubu/platform/cuda/thread_id.hpp>

namespace ns = ubu;

void test_thread_id()
{
  static_assert(ns::grid_coordinate<ns::cuda::thread_id>);
}

