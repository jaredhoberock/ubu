#include <ubu/places/causality/future/invoke_after.hpp>
#include <ubu/places/causality/past_event.hpp>
#include <ubu/places/execution/executors.hpp>
#include <ubu/places/memory/allocators.hpp>
#include <ubu/platforms/cpp/inline_executor.hpp>

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

  static ns::cpp::inline_executor associated_executor()
  {
    return {};
  }
};

static_assert(ns::allocator<trivial_asynchronous_allocator<int>>);
static_assert(ns::asynchronous_allocator<trivial_asynchronous_allocator<int>>);


void test_invoke_after()
{
  ns::cpp::inline_executor ex;
  trivial_asynchronous_allocator<int> alloc;

  auto before = ns::initial_happening(alloc);

  auto fut_13 = ubu::invoke_after(ex, alloc, before, []
  {
    int result = 13;
    return result;
  });

  auto fut_7 = ubu::invoke_after(ex, alloc, before, []
  {
    int result = 7;
    return result;
  });

  auto sum = ubu::invoke_after(ex, alloc, before, [](int a, int b)
  {
    int result = a + b;
    return result;
  }, std::move(fut_13), std::move(fut_7));

  assert(13 + 7 == sum.get());
}

