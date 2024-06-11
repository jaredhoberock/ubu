#include <cassert>
#include <iostream>
#include <ubu/tensor/coordinates/point.hpp>
#include <ubu/tensor/views/layouts/column_major.hpp>
#include <ubu/tensor/views/layouts/layout.hpp>
#include <utility>
#include <vector>

namespace ns = ubu;

void test_column_major()
{
  using tensor_t = std::vector<float>;

  static_assert(ns::layout_for<ns::column_major<int>, tensor_t>);
  static_assert(ns::layout_for<ns::column_major<ns::int2>, tensor_t>);
  static_assert(ns::layout_for<ns::column_major<ns::int3>, tensor_t>);

  using int3x4 = ns::point<ns::int4,3>;

  static_assert(ns::layout_for<ns::column_major<int3x4>, tensor_t>);
}

