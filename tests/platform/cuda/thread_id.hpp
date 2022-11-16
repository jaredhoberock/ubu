#include <ubu/coordinate/congruent.hpp>
#include <ubu/coordinate/coordinate.hpp>
#include <ubu/coordinate/detail/tuple_algorithm.hpp>
#include <ubu/coordinate/weakly_congruent.hpp>
#include <ubu/platform/cuda/thread_id.hpp>

namespace ns = ubu;

void test_thread_id()
{
  static_assert(ns::coordinate<ns::cuda::thread_id>);
  static_assert(ns::weakly_congruent<ns::cuda::thread_id, ns::cuda::thread_id>);
  static_assert(ns::congruent<ns::cuda::thread_id, ns::cuda::thread_id>);
  static_assert(ns::detail::tuple_like<ns::cuda::thread_id>);
}

