#include <ubu/platform/cuda/thread_id.hpp>
#include <ubu/tensor/coordinates/concepts/congruent.hpp>
#include <ubu/tensor/coordinates/concepts/coordinate.hpp>
#include <ubu/tensor/coordinates/concepts/weakly_congruent.hpp>
#include <ubu/tensor/coordinates/detail/tuple_algorithm.hpp>

namespace ns = ubu;

void test_thread_id()
{
  static_assert(ns::coordinate<ns::cuda::thread_id>);
  static_assert(ns::weakly_congruent<ns::cuda::thread_id, ns::cuda::thread_id>);
  static_assert(ns::congruent<ns::cuda::thread_id, ns::cuda::thread_id>);
  static_assert(ns::detail::tuple_like<ns::cuda::thread_id>);
}

