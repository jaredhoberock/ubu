#include <aspera/event/always_complete_event.hpp>
#include <aspera/execution/executor.hpp>
#include <aspera/future/invoke_after.hpp>
#include <aspera/memory/allocator.hpp>

#define NDEBUG
#include <cassert>

#include <memory>


namespace ns = aspera;

template<class T>
struct trivial_asynchronous_allocator : public std::allocator<T>
{
  using event_type = ns::always_complete_event;

  template<class U = T>
  std::pair<event_type, U*> allocate_after(const event_type& before, std::size_t n)
  {
    T* ptr = std::allocator<T>::allocate(sizeof(T) * n);
  
    return {event_type{}, ptr};
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
static_assert(ns::asynchronous_allocator<trivial_asynchronous_allocator<int>>);


void test_invoke_after()
{
  ns::inline_executor ex;
  trivial_asynchronous_allocator<int> alloc;

  auto before = ns::make_complete_event(alloc);

  auto fut_13 = aspera::invoke_after(ex, alloc, before, []
  {
    int result = 13;
    return result;
  });

  auto fut_7 = aspera::invoke_after(ex, alloc, before, []
  {
    int result = 7;
    return result;
  });

  auto sum = aspera::invoke_after(ex, alloc, before, [](int a, int b)
  {
    int result = a + b;
    return result;
  }, std::move(fut_13), std::move(fut_7));

  assert(13 + 7 == sum.get());
}

