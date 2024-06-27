#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/layouts/layout.hpp>
#include <ubu/tensors/views/layouts/row_major.hpp>
#include <vector>

namespace ns = ubu;

void test_row_major()
{
  using tensor_t = std::vector<float>;

  static_assert(ns::layout_for<ns::row_major<ns::int2>, tensor_t>);
}

