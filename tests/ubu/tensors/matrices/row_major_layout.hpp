#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/matrices/row_major_layout.hpp>
#include <ubu/tensors/views/layouts/concepts/layout_like.hpp>
#include <vector>

namespace ns = ubu;

void test_row_major_layout()
{
  using tensor_t = std::vector<float>;

  static_assert(ns::layout_like_for<ns::row_major_layout<ns::int2>, tensor_t>);
}

