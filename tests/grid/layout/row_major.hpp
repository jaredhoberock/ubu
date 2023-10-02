#include <cassert>
#include <iostream>
#include <ubu/grid/coordinate/point.hpp>
#include <ubu/grid/layout/layout.hpp>
#include <ubu/grid/layout/row_major.hpp>
#include <utility>
#include <vector>

namespace ns = ubu;

void test_row_major()
{
  using grid_t = std::vector<float>;

  static_assert(ns::layout_for<ns::row_major<int>, grid_t>);
  static_assert(ns::layout_for<ns::row_major<ns::int2>, grid_t>);
  static_assert(ns::layout_for<ns::row_major<ns::int3>, grid_t>);

  using int3x4 = ns::point<ns::int4,3>;

  static_assert(ns::layout_for<ns::row_major<int3x4>, grid_t>);
}

