#include <cassert>
#include <iostream>
#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/layout/layout.hpp>
#include <ubu/tensor/layout/row_major.hpp>
#include <utility>
#include <vector>

namespace ns = ubu;

void test_row_major()
{
  using tensor_t = std::vector<float>;

  static_assert(ns::layout_for<ns::row_major<int>, tensor_t>);
  static_assert(ns::layout_for<ns::row_major<ns::int2>, tensor_t>);
  static_assert(ns::layout_for<ns::row_major<ns::int3>, tensor_t>);

  using int3x4 = ns::point<ns::int4,3>;

  static_assert(ns::layout_for<ns::row_major<int3x4>, tensor_t>);
}

