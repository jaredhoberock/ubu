#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/layouts/compact_left_major_layout.hpp>
#include <ubu/tensors/views/layouts/concepts/layout.hpp>
#include <vector>

namespace ns = ubu;

void test_compact_left_major_layout()
{
  using tensor_t = std::vector<float>;

  static_assert(ns::layout_for<ns::compact_left_major_layout<int>, tensor_t>);
  static_assert(ns::layout_for<ns::compact_left_major_layout<ns::int2>, tensor_t>);
  static_assert(ns::layout_for<ns::compact_left_major_layout<ns::int3>, tensor_t>);

  using int3x4 = ns::point<ns::int4,3>;

  static_assert(ns::layout_for<ns::compact_left_major_layout<int3x4>, tensor_t>);
}

