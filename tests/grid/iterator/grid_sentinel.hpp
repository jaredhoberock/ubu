#include <concepts>
#include <ubu/grid/iterator.hpp>
#include <span>

namespace ns = ubu;

void test_grid_sentinel()
{
  using namespace ubu;

  static_assert(std::sentinel_for<grid_sentinel, sized_grid_iterator<std::span<int>>>);
  static_assert(std::sentinel_for<grid_sentinel, grid_iterator<std::span<int>>>);
}

