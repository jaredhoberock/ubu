#include <array>
#include <cassert>
#include <concepts>
#include <ubu/tensors/coordinates/element.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <tuple>
#include <utility>
#include <vector>


namespace ns = ubu;


struct has_element_member_function
{
  int& element(std::size_t i)
  {
    return a[i];
  }

  int a[4];
};


struct has_operator_bracket
{
  int& operator[](std::size_t i)
  {
    return a[i];
  }

  int a[4];
};


struct has_operator_parens
{
  int& operator()(std::size_t i)
  {
    return a[i];
  }

  int a[4];
};


void test_element_t()
{
  // test 2-arrays
  static_assert(std::same_as<int, ns::element_t<std::array<int, 2>, int>>);

  // test user-defined type with element member function
  static_assert(std::same_as<int, ns::element_t<has_element_member_function, int>>);

  // test user-defined type with operator[]
  static_assert(std::same_as<int, ns::element_t<has_operator_bracket, int>>);

  // test user-defined type with operator()
  static_assert(std::same_as<int, ns::element_t<has_operator_bracket, int>>);

  // test point
  static_assert(std::same_as<int, ns::element_t<ns::point<int,3>, int>>);

  // test typedefs
  static_assert(std::same_as<int, ns::element_t<ns::int2, int>>);

  // test nested std::vector with 1d coord
  static_assert(std::same_as<std::vector<int>, ns::element_t<std::vector<std::vector<int>>, int>>);

  // test nested std::vector with 2d coord
  static_assert(std::same_as<int, ns::element_t<std::vector<std::vector<int>>, ns::int2>>);
}


void test_cpo()
{
  // test 2-arrays
  {
    std::array<int, 2> x{13, 7};
    assert(13 == ns::element(x, 0));
    assert( 7 == ns::element(x, 1));
  }
  
  // test user-defined type with operator[]
  {
    has_operator_bracket x{13, 7, 42, 66};
    assert(13 == ns::element(x, 0));
    assert( 7 == ns::element(x, 1));
    assert(42 == ns::element(x, 2));
    assert(66 == ns::element(x, 3));
  }

  // test user-defined type with operator()
  {
    has_operator_parens x{13, 7, 42, 66};
    assert(13 == ns::element(x, 0));
    assert( 7 == ns::element(x, 1));
    assert(42 == ns::element(x, 2));
    assert(66 == ns::element(x, 3));
  }

  // test point
  {
    ns::point<int,3> x{13, 7, 42};
    assert(13 == ns::element(x, 0));
    assert( 7 == ns::element(x, 1));
    assert(42 == ns::element(x, 2));
  }

  // test typedefs
  {
    ns::int2 x{13, 7};
    assert(13 == ns::element(x, 0));
    assert( 7 == ns::element(x, 1));
  }

  // test nested std::vector with 1d coord
  {
    std::vector<std::vector<int>> nested_vec({{13,7}, {42,66}});
    assert(std::vector<int>({13, 7}) == ns::element(nested_vec, 0));
    assert(std::vector<int>({42,66}) == ns::element(nested_vec, 1));
  }

  // test nested std::vector with 2d coord
  {
    std::vector<std::vector<int>> nested_vec({{13,7}, {42,66}});
    assert(13 == ns::element(nested_vec, std::pair(0,0)));
    assert( 7 == ns::element(nested_vec, std::pair(1,0)));
    assert(42 == ns::element(nested_vec, std::pair(0,1)));
    assert(66 == ns::element(nested_vec, std::pair(1,1)));
  }
}


void test_element()
{
  test_cpo();
  test_element_t();
}

