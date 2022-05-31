#include <ubu/coordinate/grid_size.hpp>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

#include <tuple>
#include <utility>

void test_grid_size()
{
  namespace ns = ubu;

  // 1D spaces
  assert(13 == ns::grid_size(13));
  assert(7 == ns::grid_size(7));
  assert(13 == ns::grid_size(ns::int1(13)));
  assert(7 == ns::grid_size(ns::int1(7)));
  assert(7 == ns::grid_size(std::make_tuple(7)));

  // 2D spaces
  assert(7 * 13 == ns::grid_size(ns::int2(7,13)));
  assert(7 * 13 == ns::grid_size(std::make_tuple(7,13)));
  assert(7 * 13 == ns::grid_size(std::make_pair(7,13)));

  // 3D spaces
  assert(7 * 13 * 42 == ns::grid_size(ns::int3(7,13,42)));
  assert(7 * 13 * 42 == ns::grid_size(std::make_tuple(7,13,42)));
  assert(7 * 13 * 42 == ns::grid_size(std::array<int,3>{7,13,42}));

  // nested spaces
  assert(7 * 13 * 42 == ns::grid_size(std::make_pair(7, std::make_pair(13, 42))));
  assert(7 * 13 * 42 == ns::grid_size(std::make_pair(std::make_pair(7, 13), 42)));
  assert(7 * 13 * 42 * 123 == ns::grid_size(std::make_pair(std::make_pair(7, 13), std::make_pair(42, 123))));
  assert(7 * 13 * 42 * 123 == ns::grid_size(std::make_pair(ns::int2{7,13}, ns::int2{42,123})));
  assert(7 * 13 * 42 * 123 == ns::grid_size(std::make_pair(7, ns::int3{13,42,123})));
}

