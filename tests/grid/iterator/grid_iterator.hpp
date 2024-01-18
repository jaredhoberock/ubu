#include <concepts>
#include <ubu/grid/iterator.hpp>
#include <span>

namespace ns = ubu;

void test_grid_iterator()
{
  using namespace ubu;

  static_assert(std::random_access_iterator<grid_iterator<std::span<int>>>);
}

