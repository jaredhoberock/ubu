#include <barrier>
#include <cassert>
#include <ubu/cooperators/barriers/barrier_like.hpp>
#include <ubu/cooperators/barriers/get_barrier.hpp>

namespace ns = ubu;

struct has_barrier_member_variable
{
  std::barrier<> barrier{1};
};

struct has_get_barrier_member_function
{
  std::barrier<>& bar;

  std::barrier<>& get_barrier()
  {
    return bar;
  }
};

struct has_get_barrier_free_function
{
  std::barrier<>& bar;
};

std::barrier<>& get_barrier(has_get_barrier_free_function arg)
{
  return arg.bar;
}

void test_get_barrier()
{
  static_assert(ns::barrier_like<ubu::barrier_t<has_barrier_member_variable>>);
  static_assert(ns::barrier_like<ubu::barrier_t<has_get_barrier_member_function>>);
  static_assert(ns::barrier_like<ubu::barrier_t<has_get_barrier_free_function>>);
}

