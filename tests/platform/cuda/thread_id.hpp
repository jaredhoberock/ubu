#include <ubu/grid/coordinate/concepts/congruent.hpp>
#include <ubu/grid/coordinate/concepts/coordinate.hpp>
#include <ubu/grid/coordinate/concepts/weakly_congruent.hpp>
#include <ubu/grid/coordinate/detail/tuple_algorithm.hpp>
#include <ubu/platform/cuda/thread_id.hpp>

namespace ns = ubu;

void test_thread_id()
{
  static_assert(ns::coordinate<ns::cuda::thread_id>);
  static_assert(ns::weakly_congruent<ns::cuda::thread_id, ns::cuda::thread_id>);
  static_assert(ns::congruent<ns::cuda::thread_id, ns::cuda::thread_id>);
  static_assert(ns::detail::tuple_like<ns::cuda::thread_id>);
}

