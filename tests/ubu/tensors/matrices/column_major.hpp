#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/matrices/column_major.hpp>
#include <ubu/tensors/views/layouts/layout.hpp>
#include <vector>

namespace ns = ubu;

void test_column_major()
{
  using tensor_t = std::vector<float>;

  static_assert(ns::layout_for<ns::column_major<ns::int2>, tensor_t>);
}
