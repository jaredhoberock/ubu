#include <cassert>
#include <concepts>
#include <ubu/cooperation/barriers.hpp>

namespace ns = ubu;

struct warp_barrier
{
  static constexpr int size()
  {
    return 16;
  }

  constexpr void arrive_and_wait() const
  {
  }
};

struct cta_barrier
{
  constexpr void arrive_and_wait() const
  {
  }

  constexpr warp_barrier get_local_barrier() const
  {
    return {};
  }
};


void test_barrier_like()
{
  static_assert(ns::barrier_like<warp_barrier>);
  static_assert(ns::barrier_like<cta_barrier>);
  static_assert(ns::hierarchical_barrier_like<cta_barrier>);
}

