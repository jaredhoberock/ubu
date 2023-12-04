#include <ubu/causality/wait.hpp>
#include <ubu/execution/executor/concepts/bulk_executable_on.hpp>
#include <ubu/execution/executor/concepts/bulk_executor.hpp>
#include <ubu/execution/executor/concepts/executor.hpp>
#include <ubu/execution/executor/execute_after.hpp>
#include <ubu/execution/executor/first_execute.hpp>
#include <ubu/execution/executor/finally_execute_after.hpp>
#include <ubu/memory/buffer/reinterpret_buffer.hpp>
#include <ubu/platform/cpp/new_thread_group_executor.hpp>

#undef NDEBUG
#include <cassert>

#include <future>

namespace ns = ubu;

void test_concepts()
{
  {
    auto lambda = []{};
    static_assert(ns::executor_of<ns::new_thread_group_executor, decltype(lambda)>);
  }

  {
    auto lambda = [](auto coord, auto workspace){};
    static_assert(ns::bulk_executable_on<decltype(lambda), ns::new_thread_group_executor, std::future<void>, std::size_t, std::size_t>);
    static_assert(ns::bulk_executor<ns::new_thread_group_executor>);
  }
}

void test_single_execution()
{
  {
    bool invoked = false;

    ns::new_thread_group_executor ex;

    auto before = ns::first_cause(ex);

    auto e = ns::execute_after(ex, std::move(before), [&invoked]
    {
      invoked = true;
    });

    ns::wait(e);
    assert(invoked);
  }

  {
    bool invoked = false;

    ns::new_thread_group_executor ex;

    auto e = ns::first_execute(ex, [&invoked]
    {
      invoked = true;
    });

    ns::wait(e);
    assert(invoked);
  }

  {
    bool invoked1 = false;
    bool invoked2 = false;

    ns::new_thread_group_executor ex;

    auto e1 = ns::first_execute(ex, [&invoked1]
    {
      invoked1 = true;
    });

    auto e2 = ns::execute_after(ex, e1, [&invoked1, &invoked2]
    {
      assert(invoked1);
      invoked2 = true;
    });

    ns::wait(e2);
    assert(invoked2);
  }

  {
    bool invoked1 = false;
    bool invoked2 = false;

    ns::new_thread_group_executor ex;

    auto e1 = ns::first_execute(ex, [&invoked1]
    {
      invoked1 = true;
    });

    std::promise<void> p;
    auto f = p.get_future();
    ns::finally_execute_after(ex, e1, [p = std::move(p), &invoked1, &invoked2]() mutable
    {
      assert(invoked1);
      invoked2 = true;
      p.set_value();
    });

    f.wait();
    assert(invoked2);
  }
}

void test_bulk_execution()
{
  ns::new_thread_group_executor ex;
  std::future<void> before = ubu::first_cause(ex);

  int n = 10;
  std::atomic<int> counter = 0;

  auto after = ns::new_bulk_execute_after(ex, std::move(before), n, n * sizeof(int), [&](int coord, ns::concurrent_workspace auto workspace)
  {
    auto coords = ns::reinterpret_buffer<int>(ns::get_buffer(workspace));
    coords[coord] = coord;

    ++counter;

    ns::arrive_and_wait(ns::get_barrier(workspace));

    int expected = 0;
    for(auto coord : coords)
    {
      assert(expected == coord);
      ++expected;
    }
  });

  ns::wait(after);

  assert(n == counter);
}

void test_new_thread_group_executor()
{
  test_concepts();
  test_single_execution();
  test_bulk_execution();
}

