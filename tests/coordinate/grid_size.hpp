#include <aspera/coordinate/grid_size.hpp>
#include <aspera/coordinate/point.hpp>
#include <cassert>
#include <tuple>
#include <utility>

void test_grid_size()
{
  using namespace aspera;

  // 1D spaces
  assert(13 == grid_size(13));
  assert(7 == grid_size(7));
  assert(13 == grid_size(int1(13)));
  assert(7 == grid_size(int1(7)));
  assert(7 == grid_size(std::make_tuple(7)));

  // 2D spaces
  assert(7 * 13 == grid_size(int2(7,13)));
  assert(7 * 13 == grid_size(std::make_tuple(7,13)));
  assert(7 * 13 == grid_size(std::make_pair(7,13)));

  // 3D spaces
  assert(7 * 13 * 42 == grid_size(int3(7,13,42)));
  assert(7 * 13 * 42 == grid_size(std::make_tuple(7,13,42)));
  assert(7 * 13 * 42 == grid_size(std::array<int,3>{7,13,42}));

  // nested spaces
  assert(7 * 13 * 42 == grid_size(std::make_pair(7, std::make_pair(13, 42))));
  assert(7 * 13 * 42 == grid_size(std::make_pair(std::make_pair(7, 13), 42)));
  assert(7 * 13 * 42 * 123 == grid_size(std::make_pair(std::make_pair(7, 13), std::make_pair(42, 123))));
  assert(7 * 13 * 42 * 123 == grid_size(std::make_pair(int2{7,13}, int2{42,123})));
  assert(7 * 13 * 42 * 123 == grid_size(std::make_pair(7, int3{13,42,123})));
}

