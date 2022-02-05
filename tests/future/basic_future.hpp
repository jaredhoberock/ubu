#include <aspera/event/complete_event.hpp>
#include <aspera/executor.hpp>
#include <aspera/future/basic_future.hpp>
#include <aspera/memory/allocator.hpp>

#define NDEBUG
#include <cassert>

#include <memory>

namespace ns = aspera;

template<class T>
struct trivial_asynchronous_allocator : public std::allocator<T>
{
  using event_type = ns::complete_event;

  template<class U = T>
  ns::basic_future<U*, trivial_asynchronous_allocator> allocate_after(const event_type& before, std::size_t n)
  {
    T* ptr = std::allocator<T>::allocate(sizeof(T) * n);
  
    return {event_type{}, std::make_pair(ptr,n), *this};
  }
  
  event_type deallocate_after(const event_type&, T* ptr, std::size_t n)
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
static_assert(ns::asynchronous_deallocator<trivial_asynchronous_allocator<int>>);
static_assert(ns::asynchronous_allocator<trivial_asynchronous_allocator<int>>);


template<class T>
void test_asynchronous_allocation()
{
  trivial_asynchronous_allocator<T> alloc;

  auto future_ptr = ns::first_allocate(alloc, 1);
  
  future_ptr.wait();
}


template<class T>
void test_asynchronous_allocation_and_asynchronous_deletion()
{
  trivial_asynchronous_allocator<T> alloc;

  auto future_ptr = ns::first_allocate(alloc, 1);
  
  auto [ready, ptr, n] = std::move(future_ptr).release();
  
  assert(1 == n);
  
  auto all_done = ns::deallocate_after(alloc, ready, ptr, n);
  
  ns::wait(all_done);
}


template<class T>
void test_asynchronous_allocation_and_synchronous_deletion()
{
  trivial_asynchronous_allocator<T> alloc;

  auto future_ptr = ns::first_allocate(alloc, 1);
  
  future_ptr.wait();
  
  auto [_, ptr, n] = std::move(future_ptr).release();
  
  assert(1 == n);
  
  ns::deallocate(alloc, ptr, n);
}


template<class T>
void test_then_construct_with_no_args()
{
  trivial_asynchronous_allocator<T> alloc;

  auto future_ptr = ns::first_allocate(alloc, 1);

  ns::inline_executor ex;
  
  auto future_value = std::move(future_ptr).then_construct(ex, []
  {
    return 13;
  });
  
  assert(13 == future_value.get());
}


template<class T>
void test_then_construct_after()
{
  trivial_asynchronous_allocator<T> alloc;

  auto future_ptr = ns::first_allocate(alloc, 1);

  auto ready = ns::make_complete_event(alloc);

  ns::inline_executor ex;

  auto future_value = std::move(future_ptr).then_construct_after(ex, ready, []
  {
    return 13;
  });

  assert(13 == future_value.get());
}


template<class T>
void test_then_after()
{
  trivial_asynchronous_allocator<T> alloc;

  auto future_ptr1 = ns::first_allocate(alloc, 1);
  auto future_ptr2 = ns::first_allocate(alloc, 1);

  ns::inline_executor ex;
  
  // create one future argument
  auto future_val1 = std::move(future_ptr1).then_construct(ex, []
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = std::move(future_ptr2).then_construct(ex, []
  {
    return 7;
  });
  
  auto ready = ns::make_complete_event(alloc);
  
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
  
  auto future_ptr1 = ns::first_allocate(alloc, 1);
  auto future_ptr2 = ns::first_allocate(alloc, 1);
  
  // create one future argument
  auto future_val1 = std::move(future_ptr1).then_construct([]
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = std::move(future_ptr2).then_construct([]
  {
    return 7;
  });

  ns::inline_executor ex;
  
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

  auto future_ptr1 = ns::first_allocate(alloc, 1);
  auto future_ptr2 = ns::first_allocate(alloc, 1);
  
  // create one future argument
  auto future_val1 = std::move(future_ptr1).then_construct([]
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = std::move(future_ptr2).then_construct([]
  {
    return 7;
  });

  ns::inline_executor ex;
  
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

  auto future_ptr1 = ns::first_allocate(alloc, 1);
  auto future_ptr2 = ns::first_allocate(alloc, 1);
  
  // create one future argument
  auto future_val1 = std::move(future_ptr1).then_construct([]
  {
    return 13;
  });
  
  // create a second future argument
  auto future_val2 = std::move(future_ptr2).then_construct([]
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


void test_basic_future()
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

  test_then_construct_with_no_args<char>();
  test_then_construct_with_no_args<int>();
  test_then_construct_with_no_args<double>();

  test_then_construct_after<char>();
  test_then_construct_after<int>();
  test_then_construct_after<double>();

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

