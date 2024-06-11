#include <concepts>
#include <ubu/tensors/iterators.hpp>
#include <span>

namespace ns = ubu;

void test_sized_tensor_iterator()
{
  using namespace ubu;

  static_assert(std::random_access_iterator<sized_tensor_iterator<std::span<int>>>);
}

