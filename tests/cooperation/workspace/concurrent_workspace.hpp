#include <span>
#include <ubu/cooperation/workspace/workspace.hpp>

namespace ns = ubu;

struct has_buffer_member_variable
{
  const std::span<std::byte> buffer = {};
};

struct has_get_buffer_member_function
{
  std::span<std::byte> get_buffer()
  {
    return {};
  }
};

struct has_get_buffer_free_function
{
};

std::span<std::byte> get_buffer(has_get_buffer_free_function)
{
  return {};
}

struct barrier_like
{
  void arrive_and_wait();
};

template<ns::buffer_like B>
struct has_barrier_member_variable
{
  B buffer;
  barrier_like barrier;
};

template<ns::buffer_like B>
struct has_get_barrier_member_function
{
  B buffer;
  barrier_like get_barrier();
};

template<ns::buffer_like B>
struct has_get_barrier_free_function
{
  B buffer;
};

template<ns::buffer_like B>
barrier_like get_barrier(has_get_barrier_free_function<B>)
{
  return {};
}

void test_concurrent_workspace()
{
  // a std::span of bytes is a workspace
  static_assert(ns::workspace<std::span<std::byte>>);

  // a thing with a buffer is a workspace
  static_assert(ns::workspace<has_buffer_member_variable>);
  static_assert(ns::workspace<has_get_buffer_member_function>);
  static_assert(ns::workspace<has_get_buffer_free_function>);

  // a thing with a buffer and barrier is a concurrent_workspace
  static_assert(ns::concurrent_workspace<has_barrier_member_variable<std::span<std::byte>>>);
  static_assert(ns::concurrent_workspace<has_get_barrier_member_function<std::span<std::byte>>>);
  static_assert(ns::concurrent_workspace<has_get_barrier_free_function<std::span<std::byte>>>);
}

