#include <memory>
#include <ubu/places/memory/allocators/concepts/allocator.hpp>

void test_concepts()
{
  using namespace ubu;

  static_assert(allocator<std::allocator<int>>);
}

