#include <ubu/causality/wait.hpp>
#include <ubu/execution/executor/concepts/bulk_executable_on.hpp>
#include <ubu/execution/executor/concepts/bulk_executable_with_workspace_on.hpp>
#include <ubu/execution/executor/concepts/bulk_executor.hpp>
#include <ubu/execution/executor/concepts/executor.hpp>
#include <ubu/execution/executor/execute_after.hpp>
#include <ubu/execution/executor/first_execute.hpp>
#include <ubu/execution/executor/finally_execute_after.hpp>
#include <ubu/memory/buffer/reinterpret_buffer.hpp>
#include <ubu/platform/cpp/new_thread_group_executor.hpp>
#include <barrier>
#include <numeric>
#include <cstddef>
#include <memory>
#include <vector>

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
    auto lambda = [](auto coord){};
    static_assert(ns::bulk_executable_on<decltype(lambda), ns::new_thread_group_executor, std::future<void>, std::size_t>);
  }

  {
    auto lambda = [](auto coord, auto workspace){};
    static_assert(ns::bulk_executable_with_workspace_on<decltype(lambda), ns::new_thread_group_executor, std::allocator<std::byte>, std::future<void>, std::size_t, std::size_t>);
  }

  {
    static_assert(ns::bulk_executor<ns::new_thread_group_executor>);
  }
}

void test_single_execution()
{
  {
    bool invoked = false;

    ns::new_thread_group_executor ex;

    auto before = ns::initial_happening(ex);

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

void test_bulk_execute_after_cpo()
{
  ns::new_thread_group_executor ex;
  std::future<void> before = ubu::initial_happening(ex);

  int n = 10;

  std::vector<int> expected(n);
  std::iota(expected.begin(), expected.end(), 0);

  std::vector<int> arrived(n, -1);

  std::barrier<> barrier(n);
  auto after = ns::bulk_execute_after(ex, std::move(before), n, [&](int coord)
  {
    // make sure each thread has concurrent fwd progress
    barrier.arrive_and_wait();

    arrived[coord] = coord;
  });

  ns::wait(after);

  assert(expected == arrived);
}

void test_bulk_execute_with_workspace_after_member()
{
  ns::new_thread_group_executor ex;
  std::future<void> before = ubu::initial_happening(ex);

  int n = 10;

  auto after = ex.bulk_execute_with_workspace_after(std::move(before), n, n * sizeof(int), [&](int coord, ns::concurrent_workspace auto workspace)
  {
    auto coords = ns::reinterpret_buffer<int>(ns::get_buffer(workspace));
    coords[coord] = coord;

    ns::arrive_and_wait(ns::get_barrier(workspace));

    int expected = 0;
    for(auto coord : coords)
    {
      assert(expected == coord);
      ++expected;
    }

    assert(n == expected);
  });

  ns::wait(after);
}

void test_bulk_execute_with_workspace_after_cpo()
{
  std::allocator<std::byte> alloc;
  ns::new_thread_group_executor ex;
  std::future<void> before = ubu::initial_happening(ex);

  int n = 10;

  auto after = ns::bulk_execute_with_workspace_after(ex, alloc, std::move(before), n, n * sizeof(int), [&](int coord, ns::concurrent_workspace auto workspace)
  {
    auto coords = ns::reinterpret_buffer<int>(ns::get_buffer(workspace));
    coords[coord] = coord;

    ns::arrive_and_wait(ns::get_barrier(workspace));

    int expected = 0;
    for(auto coord : coords)
    {
      assert(expected == coord);
      ++expected;
    }

    assert(n == expected);
  });

  ns::wait(after);
}

void test_bulk_execution()
{
  test_bulk_execute_after_cpo();

  test_bulk_execute_with_workspace_after_member();
  test_bulk_execute_with_workspace_after_cpo();
}

void test_new_thread_group_executor()
{
  test_concepts();
  test_single_execution();
  test_bulk_execution();
}

