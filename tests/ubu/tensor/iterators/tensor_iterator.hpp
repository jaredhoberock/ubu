#include <concepts>
#include <ubu/tensor/iterators.hpp>
#include <span>

namespace ns = ubu;

void test_tensor_iterator()
{
  using namespace ubu;

  static_assert(std::random_access_iterator<tensor_iterator<std::span<int>>>);
}

