#include <array>
#include <concepts>
#include <ubu/miscellaneous/constant.hpp>
#include <ubu/tensors/coordinates/element.hpp>
#include <ubu/tensors/coordinates/point.hpp>

#undef NDEBUG
#include <cassert>

#include <tuple>
#include <utility>


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
  using namespace ns;

  // test size-1 coordinates
  static_assert(std::same_as<int, ns::element_t<int, constant<0>>>);
  static_assert(std::same_as<unsigned int, ns::element_t<unsigned int, constant<0>>>);
  static_assert(std::same_as<std::size_t, ns::element_t<std::size_t, constant<0>>>);
  static_assert(std::same_as<char, ns::element_t<std::tuple<char>, constant<0>>>);

  // test homogeneous pairs
  static_assert(std::same_as<int, ns::element_t<std::pair<int,int>, constant<0>>>);
  static_assert(std::same_as<int, ns::element_t<std::pair<int,int>, constant<1>>>);

  // test heterogeneous 2-tuples
  static_assert(std::same_as<int, ns::element_t<std::tuple<int, unsigned int>, constant<0>>>);
  static_assert(std::same_as<unsigned int, ns::element_t<std::tuple<int, unsigned int>, constant<1>>>);

  // test 2-arrays
  static_assert(std::same_as<int, ns::element_t<std::array<int, 2>, constant<0>>>);
  static_assert(std::same_as<int, ns::element_t<std::array<int, 2>, constant<1>>>);
  
  // test heterogeneous 3-tuples
  static_assert(std::same_as<int, ns::element_t<std::tuple<int,unsigned int,std::size_t>, constant<0>>>);
  static_assert(std::same_as<unsigned int, ns::element_t<std::tuple<int,unsigned int,std::size_t>, constant<1>>>);
  static_assert(std::same_as<std::size_t, ns::element_t<std::tuple<int,unsigned int,std::size_t>, constant<2>>>);

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

  // test heterogeneous pairs of homogeneous coordinates
  static_assert(std::same_as< int2, ns::element_t<std::pair<int2,uint3>, constant<0>>>);
  static_assert(std::same_as<uint3, ns::element_t<std::pair<int2,uint3>, constant<1>>>);

  // test heterogeneous tuples of homogeneous coordinates
  static_assert(std::same_as<       int2, ns::element_t<std::tuple<int2,uint3,std::size_t>, constant<0>>>);
  static_assert(std::same_as<      uint3, ns::element_t<std::tuple<int2,uint3,std::size_t>, constant<1>>>);
  static_assert(std::same_as<std::size_t, ns::element_t<std::tuple<int2,uint3,std::size_t>, constant<2>>>);

  // test another level of tupling
  using uint2x3 = std::pair<uint3,uint3>;
  static_assert(std::same_as<uint3, ns::element_t<uint2x3, constant<0>>>);
  static_assert(std::same_as<uint3, ns::element_t<uint2x3, constant<1>>>);

  // test user-defined type with element member function
  static_assert(std::same_as<int, ns::element_t<has_element_member_function, constant<0>>>);
  static_assert(std::same_as<int, ns::element_t<has_element_member_function, constant<1>>>);
  static_assert(std::same_as<int, ns::element_t<has_element_member_function, constant<2>>>);
  static_assert(std::same_as<int, ns::element_t<has_element_member_function, constant<3>>>);

  // test user-defined type with operator[]
  static_assert(std::same_as<int, ns::element_t<has_operator_bracket, constant<0>>>);
  static_assert(std::same_as<int, ns::element_t<has_operator_bracket, constant<1>>>);
  static_assert(std::same_as<int, ns::element_t<has_operator_bracket, constant<2>>>);
  static_assert(std::same_as<int, ns::element_t<has_operator_bracket, constant<3>>>);

  // test user-defined type with operator()
  static_assert(std::same_as<int, ns::element_t<has_operator_parens, constant<0>>>);
  static_assert(std::same_as<int, ns::element_t<has_operator_parens, constant<1>>>);
  static_assert(std::same_as<int, ns::element_t<has_operator_parens, constant<2>>>);
  static_assert(std::same_as<int, ns::element_t<has_operator_parens, constant<3>>>);

  // test point
  static_assert(std::same_as<int, ns::element_t<ns::point<int,3>, constant<0>>>);
  static_assert(std::same_as<int, ns::element_t<ns::point<int,3>, constant<1>>>);
  static_assert(std::same_as<int, ns::element_t<ns::point<int,3>, constant<2>>>);

  // test typedefs
  static_assert(std::same_as<int, ns::element_t<ns::int2, constant<0>>>);
  static_assert(std::same_as<int, ns::element_t<ns::int2, constant<1>>>);
}


void test_element_cpo()
{
  using namespace ns;

  // test integral types
  {              int x{13}; assert(13 == ns::element(x, 0_c)); }
  {     unsigned int x{13}; assert(13 == ns::element(x, 0_c)); }
  {      std::size_t x{13}; assert(13 == ns::element(x, 0_c)); }
  { std::tuple<char> x{13}; assert(13 == ns::element(x, 0_c)); }

  // test homogeneous pairs
  {
    std::pair<int,int> x{13,7};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
  }

  // test heterogeneous 2-tuples
  {
    std::tuple<int, unsigned int> x{13, 7};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
  }

  // test 2-arrays
  {
    std::array<int, 2> x{13, 7};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
  }
  
  // test heterogeneous 3-tuples
  {
    std::tuple<int, unsigned int, std::size_t> x{13, 7, 42};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
    assert(42 == ns::element(x, 2_c));
  }

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

  // test heterogeneous pairs of homogeneous coordinates
  {
    std::pair<int2,uint3> x{{13,7}, {0,1,2}};
    assert((int2{13,7} == ns::element(x, 0_c)));
    assert((uint3{0,1,2} == ns::element(x, 1_c)));
  }

  // test heterogeneous tuples of homogeneous coordinates
  {
    std::tuple<int2,uint3,std::size_t> x{{13,7}, {0,1,2}, 42};
    assert((int2{13,7} == ns::element(x, 0_c)));
    assert((uint3{0,1,2} == ns::element(x, 1_c)));
    assert((size_t{42} == ns::element(x, 2_c)));
  }

  // test another level of tupling
  using uint2x3 = std::pair<uint3,uint3>;
  {
    uint2x3 x{{0,2,4}, {1,3,5}};
    assert((uint3{0,2,4} == ns::element(x, 0_c)));
    assert((uint3{1,3,5} == ns::element(x, 1_c)));
  }

  // test user-defined type with element member function
  {
    has_element_member_function x{13, 7, 42, 66};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
    assert(42 == ns::element(x, 2_c));
    assert(66 == ns::element(x, 3_c));
  }

  // test user-defined type with operator[]
  {
    has_operator_bracket x{13, 7, 42, 66};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
    assert(42 == ns::element(x, 2_c));
    assert(66 == ns::element(x, 3_c));
  }

  // test point
  {
    ns::point<int,3> x{13, 7, 42};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
    assert(42 == ns::element(x, 2_c));
  }

  // test typedefs
  {
    ns::int2 x{13, 7};
    assert(13 == ns::element(x, 0_c));
    assert( 7 == ns::element(x, 1_c));
  }
}


void test_element()
{
  test_element_cpo();
  test_element_t();
}

