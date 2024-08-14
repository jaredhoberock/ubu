#include <ubu/utilities/constant.hpp>
#include <ubu/tensors/coordinates/concepts/bounded_coordinate.hpp>
#include <tuple>

void test_bounded_coordinate()
{
  using namespace ubu;

  static_assert(bounded_coordinate<std::tuple<>>);
  static_assert(bounded_coordinate<constant<1>>);
}

