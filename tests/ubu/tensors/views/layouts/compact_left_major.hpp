#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/layouts/compact_left_major.hpp>
#include <ubu/tensors/views/layouts/layout.hpp>
#include <vector>

namespace ns = ubu;

void test_compact_left_major()
{
  using tensor_t = std::vector<float>;

  static_assert(ns::layout_for<ns::compact_left_major<int>, tensor_t>);
  static_assert(ns::layout_for<ns::compact_left_major<ns::int2>, tensor_t>);
  static_assert(ns::layout_for<ns::compact_left_major<ns::int3>, tensor_t>);

  using int3x4 = ns::point<ns::int4,3>;

  static_assert(ns::layout_for<ns::compact_left_major<int3x4>, tensor_t>);
}

