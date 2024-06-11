#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/shapes/shape_size.hpp>

#undef NDEBUG
#include <cassert>

#include <tuple>
#include <utility>

void test_shape_size()
{
  namespace ns = ubu;

  // 1D spaces
  assert(13 == ns::shape_size(13));
  assert(7 == ns::shape_size(7));
  assert(13 == ns::shape_size(ns::int1(13)));
  assert(7 == ns::shape_size(ns::int1(7)));
  assert(7 == ns::shape_size(std::make_tuple(7)));

  // 2D spaces
  assert(7 * 13 == ns::shape_size(ns::int2(7,13)));
  assert(7 * 13 == ns::shape_size(std::make_tuple(7,13)));
  assert(7 * 13 == ns::shape_size(std::make_pair(7,13)));

  // 3D spaces
  assert(7 * 13 * 42 == ns::shape_size(ns::int3(7,13,42)));
  assert(7 * 13 * 42 == ns::shape_size(std::make_tuple(7,13,42)));
  assert(7 * 13 * 42 == ns::shape_size(std::array<int,3>{7,13,42}));

  // nested spaces
  assert(7 * 13 * 42 == ns::shape_size(std::make_pair(7, std::make_pair(13, 42))));
  assert(7 * 13 * 42 == ns::shape_size(std::make_pair(std::make_pair(7, 13), 42)));
  assert(7 * 13 * 42 * 123 == ns::shape_size(std::make_pair(std::make_pair(7, 13), std::make_pair(42, 123))));
  assert(7 * 13 * 42 * 123 == ns::shape_size(std::make_pair(ns::int2{7,13}, ns::int2{42,123})));
  assert(7 * 13 * 42 * 123 == ns::shape_size(std::make_pair(7, ns::int3{13,42,123})));
}

