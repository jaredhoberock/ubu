#include <span>
#include <ubu/cooperation/workspaces/workspace.hpp>

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

void test_workspace()
{
  // a std::span of bytes is a workspace
  static_assert(ns::workspace<std::span<std::byte>>);

  // a thing with a buffer is a workspace
  static_assert(ns::workspace<has_buffer_member_variable>);
  static_assert(ns::workspace<has_get_buffer_member_function>);
  static_assert(ns::workspace<has_get_buffer_free_function>);
}

