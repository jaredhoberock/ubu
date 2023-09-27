#include <array>
#include <concepts>
#include <string>
#include <tuple>
#include <ubu/grid/grid.hpp>
#include <ubu/grid/lattice.hpp>
#include <utility>
#include <vector>

namespace ns = ubu;

template<class Grid, class Element, class Shape>
void test_should_be_a_grid()
{
  using namespace ns;

  static_assert(grid<Grid>);
  static_assert(grid<Grid&>);
  static_assert(grid<const Grid&>);

  static_assert(std::same_as<Element, grid_element_t<Grid>>);
  static_assert(std::same_as<Element, grid_element_t<Grid&>>);
  static_assert(std::same_as<Element, grid_element_t<const Grid&>>);

  static_assert(std::same_as<Shape, grid_shape_t<Grid>>);
  static_assert(std::same_as<Shape, grid_shape_t<Grid&>>);
  static_assert(std::same_as<Shape, grid_shape_t<const Grid&>>);

  static_assert(grid_of<Grid, Element>);
  static_assert(grid_of<Grid&, Element>);
  static_assert(grid_of<const Grid&, Element>);
}

template<class T>
void test_non_grid()
{
  static_assert(not ns::grid<T>);
}

void test_grid()
{
  // test some grids
  test_should_be_a_grid<std::vector<int>, int, std::size_t>();
  test_should_be_a_grid<std::vector<float>, float, std::size_t>();
  test_should_be_a_grid<std::array<int, 4>, int, std::size_t>();
  test_should_be_a_grid<std::string, char, std::size_t>();
  test_should_be_a_grid<ns::lattice<int>, int, int>();
  test_should_be_a_grid<ns::lattice<ns::int2>, ns::int2, ns::int2>();

  // test some non grids
  test_non_grid<std::tuple<int,int>>();
  test_non_grid<std::pair<int,float>>();
  test_non_grid<int>();
}

