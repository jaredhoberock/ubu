#include <cassert>
#include <concepts>
#include <ubu/cooperation/barrier.hpp>
#include <ubu/cooperation/cooperator/basic_cooperator.hpp>

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

static_assert(ns::barrier_like<warp_barrier>);

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

static_assert(ns::barrier_like<cta_barrier>);

using warp  = ns::basic_cooperator<warp_barrier>;
static_assert(ns::cooperator<warp>);

using block = ns::basic_cooperator<cta_barrier>;
static_assert(ns::cooperator<block>);
static_assert(ns::hierarchical_cooperator<block>);
static_assert(std::same_as<warp, ns::child_cooperator_t<block>>);


void test_basic_cooperator()
{
  {
    block self(0, 32*32, cta_barrier());

    auto [w, warp_id] = ns::descend_with_group_coord(self);
    static_assert(std::same_as<warp, decltype(w)>);
    assert(warp_id == 0);

    // descend(self) should return warp
    [[maybe_unused]] warp w1 = ns::descend(self);

    // a warp should be directly constructible from a block
    [[maybe_unused]] auto w2 = warp(self);
  }

  {
    using block2d = ns::basic_cooperator<cta_barrier, ns::int2>;

    block2d self(ns::int2(0,0), ns::int2(32,32), cta_barrier());

    auto [w, warp_id] = ns::descend_with_group_coord(self);
    static_assert(std::same_as<warp, decltype(w)>);
    assert(warp_id == 0);

    // descend(self) should return warp
    [[maybe_unused]] warp w1 = ns::descend(self);

    // a warp should be directly constructible from a block
    [[maybe_unused]] auto w2 = warp(self);
  }
}

