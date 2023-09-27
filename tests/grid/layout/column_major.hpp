#include <cassert>
#include <iostream>
#include <ubu/grid/coordinate/point.hpp>
#include <ubu/grid/layout/column_major.hpp>
#include <ubu/grid/layout/layout.hpp>
#include <utility>

namespace ns = ubu;

void test_column_major()
{
  static_assert(ns::layout_onto<ns::column_major<int>, int, int>);
  static_assert(ns::layout_onto<ns::column_major<ns::int2>, ns::int2, int>);
  static_assert(ns::layout_onto<ns::column_major<ns::int3>, ns::int3, int>);

  using int3x4 = ns::point<ns::int4,3>;

  static_assert(ns::layout_onto<ns::column_major<int3x4>, ns::int3, int>);
}

