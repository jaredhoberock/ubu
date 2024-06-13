#include <cassert>
#include <cstddef>
#include <span>
#include <ubu/places/memory/buffers/get_buffer.hpp>

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

void test_get_buffer()
{
  assert(ns::get_buffer(std::span<std::byte>()).size() == 0);
  assert(ns::get_buffer(has_buffer_member_variable{}).size() == 0);
  assert(ns::get_buffer(has_get_buffer_member_function{}).size() == 0);
  assert(ns::get_buffer(has_get_buffer_free_function{}).size() == 0);
}

