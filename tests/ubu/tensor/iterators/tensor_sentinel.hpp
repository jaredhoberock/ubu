#include <concepts>
#include <ubu/tensor/iterators.hpp>
#include <span>

namespace ns = ubu;

void test_tensor_sentinel()
{
  using namespace ubu;

  static_assert(std::sentinel_for<tensor_sentinel, sized_tensor_iterator<std::span<int>>>);
  static_assert(std::sentinel_for<tensor_sentinel, tensor_iterator<std::span<int>>>);
}

