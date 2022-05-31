#include <ubu/execution/executor/execute.hpp>
#include <ubu/execution/executor/execute_after.hpp>
#include <ubu/execution/executor/executor_of.hpp>
#include <ubu/execution/executor/finally_execute_after.hpp>
#include <ubu/execution/executor/first_execute.hpp>
#include <ubu/execution/executor/new_thread_executor.hpp>

#undef NDEBUG
#include <cassert>

#include <future>

namespace ns = ubu;


void test_new_thread_executor()
{
  {
    auto lambda = []{};
    static_assert(ns::executor_of<ns::new_thread_executor, decltype(lambda)>);
  }

  {
    bool invoked = false;

    ns::new_thread_executor ex;

    std::promise<void> p;
    auto f = p.get_future();
    ns::execute(ex, [p = std::move(p), &invoked]() mutable
    {
      invoked = true;
      p.set_value();
    });

    f.wait();
    assert(invoked);
  }

  {
    bool invoked = false;

    ns::new_thread_executor ex;

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

    ns::new_thread_executor ex;

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

    ns::new_thread_executor ex;

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

