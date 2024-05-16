#include <ranges>
#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/iterator.hpp>
#include <ubu/tensor/lattice.hpp>
#include <span>

void test_enumerated_tensor_iterator()
{
  using namespace ubu;

  {
    std::span<int> tensor;

    enumerated_tensor_iterator iter(tensor);

    static_assert(std::random_access_iterator<decltype(iter)>);
  }

  {
    lattice<ubu::int3> tensor;

    enumerated_tensor_iterator iter(tensor);

    static_assert(std::random_access_iterator<decltype(iter)>);
  }
}

