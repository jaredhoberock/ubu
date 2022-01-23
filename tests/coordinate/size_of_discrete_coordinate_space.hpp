#include <aspera/coordinate/point.hpp>
#include <aspera/coordinate/size_of_discrete_coordinate_space.hpp>
#include <cassert>
#include <tuple>
#include <utility>

void test_size_of_discrete_coordinate_space()
{
  using namespace aspera;

  // 1D spaces
  assert(13 == size_of_discrete_coordinate_space(13));
  assert(7 == size_of_discrete_coordinate_space(7));
  assert(13 == size_of_discrete_coordinate_space(int1(13)));
  assert(7 == size_of_discrete_coordinate_space(int1(7)));
  assert(7 == size_of_discrete_coordinate_space(std::make_tuple(7)));

  // 2D spaces
  assert(7 * 13 == size_of_discrete_coordinate_space(int2(7,13)));
  assert(7 * 13 == size_of_discrete_coordinate_space(std::make_tuple(7,13)));
  assert(7 * 13 == size_of_discrete_coordinate_space(std::make_pair(7,13)));

  // 3D spaces
  assert(7 * 13 * 42 == size_of_discrete_coordinate_space(int3(7,13,42)));
  assert(7 * 13 * 42 == size_of_discrete_coordinate_space(std::make_tuple(7,13,42)));
  assert(7 * 13 * 42 == size_of_discrete_coordinate_space(std::array<int,3>{7,13,42}));

  // nested spaces
  assert(7 * 13 * 42 == size_of_discrete_coordinate_space(std::make_pair(7, std::make_pair(13, 42))));
  assert(7 * 13 * 42 == size_of_discrete_coordinate_space(std::make_pair(std::make_pair(7, 13), 42)));
  assert(7 * 13 * 42 * 123 == size_of_discrete_coordinate_space(std::make_pair(std::make_pair(7, 13), std::make_pair(42, 123))));
  assert(7 * 13 * 42 * 123 == size_of_discrete_coordinate_space(std::make_pair(int2{7,13}, int2{42,123})));
  assert(7 * 13 * 42 * 123 == size_of_discrete_coordinate_space(std::make_pair(7, int3{13,42,123})));
}

