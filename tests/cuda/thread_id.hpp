#include <aspera/coordinate/grid_coordinate.hpp>
#include <aspera/cuda/thread_id.hpp>

namespace ns = aspera;

void test_thread_id()
{
  static_assert(ns::grid_coordinate<ns::cuda::thread_id>);
}

