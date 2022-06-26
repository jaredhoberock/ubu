#include <ubu/causality/past_event.hpp>
#include <ubu/execution/executor.hpp>
#include <ubu/future/intrusive_future.hpp>
#include <ubu/future/invoke_after.hpp>
#include <ubu/memory/allocator.hpp>

#define NDEBUG
#include <cassert>

#include <memory>

namespace ns = ubu;

template<class T>
struct trivial_asynchronous_allocator : public std::allocator<T>
{
  using happening_type = ns::past_event;

  std::pair<ns::past_event, T*> allocate_after(const ns::past_event& before, std::size_t n)
  {
    T* ptr = std::allocator<T>::allocate(sizeof(T) * n);
  
    return {{}, ptr};
  }
  
  ns::past_event deallocate_after(const ns::past_event&, T* ptr, std::size_t n)
  {
    std::allocator<T>::deallocate(ptr, sizeof(T) * n);
    return {};
  }

  static ns::inline_executor associated_executor()
  {
    return {};
  }
};

static_assert(ns::allocator<trivial_asynchronous_allocator<int>>);
static_assert(ns::asynchronous_allocator<trivial_asynchronous_allocator<int>>);


template<class T>
void test_asynchronous_allocation()
{
  trivial_asynchronous_allocator<T> alloc;

  auto [ready,ptr] = ns::first_allocate<T>(alloc, 1);

  ready.wait();

  ns::deallocate(alloc, ptr, 1);
}


template<class T>
void test_asynchronous_allocation_and_asynchronous_deletion()
{
  trivial_asynchronous_allocator<T> alloc;

  auto [ready, ptr] = ns::first_allocate<T>(alloc, 1);
  
  auto all_done = ns::deallocate_after(alloc, ready, ptr, 1);
  
  ns::wait(all_done);
}


template<class T>
void test_asynchronous_allocation_and_synchronous_deletion()
{
  trivial_asynchronous_allocator<T> alloc;

  auto [ready,ptr] = ns::first_allocate<T>(alloc, 1);
  
  ns::deallocate(alloc, ptr, 1);
}


template<class T>
void test_then_after()
{
  trivial_asynchronous_allocator<T> alloc;
  ns::inline_executor ex;

  auto before = ns::first_cause(alloc);
  
  // create one future argument
  auto future_val1 = ns::invoke_after(ex, alloc, before, []
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = ns::invoke_after(ex, alloc, before, []
  {
    return 7;
  });
  
  auto ready = ns::first_cause(alloc);
  
  // then invoke this lambda when both arguments are ready
  auto future_val3 = std::move(future_val1).then_after(ex, alloc, std::move(ready), [](T&& arg1, T&& arg2)
  {
    return arg1 + arg2;
  }, std::move(future_val2));
  
  assert(13 + 7 == future_val3.get());
}


template<class T>
void test_then_with_allocator()
{
  trivial_asynchronous_allocator<T> alloc;
  ns::inline_executor ex;

  auto before = ns::first_cause(alloc);
  
  // create one future argument
  auto future_val1 = ns::invoke_after(ex, alloc, before, []
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = ns::invoke_after(ex, alloc, before, []
  {
    return 7;
  });
  
  // then invoke this lambda when both arguments are ready
  auto future_val3 = std::move(future_val1).then(ex, alloc, [](T&& arg1, T&& arg2)
  {
    return arg1 + arg2;
  }, std::move(future_val2));
  
  assert(13 + 7 == future_val3.get());
}


template<class T>
void test_then_with_executor()
{
  trivial_asynchronous_allocator<T> alloc;
  ns::inline_executor ex;

  auto before = ns::first_cause(alloc);
  
  // create one future argument
  auto future_val1 = ns::invoke_after(ex, alloc, before, []
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = ns::invoke_after(ex, alloc, before, []
  {
    return 7;
  });
  
  // then invoke this lambda when both arguments are ready
  auto future_val3 = std::move(future_val1).then(ex, [](T&& arg1, T&& arg2)
  {
    return arg1 + arg2;
  }, std::move(future_val2));
  
  assert(13 + 7 == future_val3.get());
}


template<class T>
void test_then()
{
  trivial_asynchronous_allocator<T> alloc;
  ns::inline_executor ex;

  auto before = ns::first_cause(alloc);
  
  // create one future argument
  auto future_val1 = ns::invoke_after(ex, alloc, before, []
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = ns::invoke_after(ex, alloc, before, []
  {
    return 7;
  });
  
  // then invoke this lambda when both arguments are ready
  auto future_val3 = std::move(future_val1).then([](T&& arg1, T&& arg2)
  {
    return arg1 + arg2;
  }, std::move(future_val2));
  
  assert(13 + 7 == future_val3.get());
}


void test_intrusive_future()
{
  test_asynchronous_allocation<char>();
  test_asynchronous_allocation<int>();
  test_asynchronous_allocation<double>();

  test_asynchronous_allocation_and_asynchronous_deletion<char>();
  test_asynchronous_allocation_and_asynchronous_deletion<int>();
  test_asynchronous_allocation_and_asynchronous_deletion<double>();

  test_asynchronous_allocation_and_synchronous_deletion<char>();
  test_asynchronous_allocation_and_synchronous_deletion<int>();
  test_asynchronous_allocation_and_synchronous_deletion<double>();

  test_then_after<char>();
  test_then_after<int>();
  test_then_after<double>();

  test_then_with_allocator<char>();
  test_then_with_allocator<int>();
  test_then_with_allocator<double>();

  test_then_with_executor<char>();
  test_then_with_executor<int>();
  test_then_with_executor<double>();

  test_then<char>();
  test_then<int>();
  test_then<double>();
}

