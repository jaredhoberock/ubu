#include <array>
#include <ubu/grid/coordinate/element.hpp>
#include <ubu/grid/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

#include <tuple>
#include <utility>


namespace ns = ubu;


struct has_element_member_function
{
  constexpr std::size_t rank()
  {
    return 4;
  }

  template<std::size_t i>
  int& element()
  {
    return a[i];
  }

  int a[4];
};


struct has_operator_bracket
{
  constexpr std::size_t rank()
  {
    return 4;
  }

  int& operator[](std::size_t i)
  {
    return a[i];
  }

  int a[4];
};


void test_element_t()
{
  // test size-1 coordinates
  static_assert(std::is_same<int, ns::element_t<0, int>>::value);
  static_assert(std::is_same<unsigned int, ns::element_t<0, unsigned int>>::value);
  static_assert(std::is_same<std::size_t, ns::element_t<0, std::size_t>>::value);
  static_assert(std::is_same<char, ns::element_t<0, std::tuple<char>>>::value);

  // test homogeneous pairs
  static_assert(std::is_same<int, ns::element_t<0, std::pair<int,int>>>::value);
  static_assert(std::is_same<int, ns::element_t<1, std::pair<int,int>>>::value);

  // test heterogeneous 2-tuples
  static_assert(std::is_same<int, ns::element_t<0, std::tuple<int, unsigned int>>>::value);
  static_assert(std::is_same<unsigned int, ns::element_t<1, std::tuple<int, unsigned int>>>::value);

  // test 2-arrays
  static_assert(std::is_same<int, ns::element_t<0, std::array<int, 2>>>::value);
  static_assert(std::is_same<int, ns::element_t<1, std::array<int, 2>>>::value);
  
  // test heterogeneous 3-tuples
  static_assert(std::is_same<int, ns::element_t<0, std::tuple<int,unsigned int,std::size_t>>>::value);
  static_assert(std::is_same<unsigned int, ns::element_t<1, std::tuple<int,unsigned int,std::size_t>>>::value);
  static_assert(std::is_same<std::size_t, ns::element_t<2, std::tuple<int,unsigned int,std::size_t>>>::value);

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

  // test heterogeneous pairs of homogeneous coordinates
  static_assert(std::is_same<int2, ns::element_t<0, std::pair<int2,uint3>>>::value);
  static_assert(std::is_same<uint3, ns::element_t<1, std::pair<int2,uint3>>>::value);

  // test heterogeneous tuples of homogeneous coordinates
  static_assert(std::is_same<int2, ns::element_t<0, std::tuple<int2,uint3,std::size_t>>>::value);
  static_assert(std::is_same<uint3, ns::element_t<1, std::tuple<int2,uint3,std::size_t>>>::value);
  static_assert(std::is_same<std::size_t, ns::element_t<2, std::tuple<int2,uint3,std::size_t>>>::value);

  // test another level of tupling
  using uint2x3 = std::pair<uint3,uint3>;
  static_assert(std::is_same<uint3, ns::element_t<0, uint2x3>>::value);
  static_assert(std::is_same<uint3, ns::element_t<1, uint2x3>>::value);

  // test user-defined type with element member function
  static_assert(std::is_same<int, ns::element_t<0, has_element_member_function>>::value);
  static_assert(std::is_same<int, ns::element_t<1, has_element_member_function>>::value);
  static_assert(std::is_same<int, ns::element_t<2, has_element_member_function>>::value);
  static_assert(std::is_same<int, ns::element_t<3, has_element_member_function>>::value);

  // test user-defined type with operator[]
  static_assert(std::is_same<int, ns::element_t<0, has_operator_bracket>>::value);
  static_assert(std::is_same<int, ns::element_t<1, has_operator_bracket>>::value);
  static_assert(std::is_same<int, ns::element_t<2, has_operator_bracket>>::value);
  static_assert(std::is_same<int, ns::element_t<3, has_operator_bracket>>::value);

  // test point
  static_assert(std::is_same<int, ns::element_t<0, ns::point<int,3>>>::value, "Error.");
  static_assert(std::is_same<int, ns::element_t<1, ns::point<int,3>>>::value, "Error.");
  static_assert(std::is_same<int, ns::element_t<2, ns::point<int,3>>>::value, "Error.");

  // test typedefs
  static_assert(std::is_same<int, ns::element_t<0, ns::int2>>::value, "Error.");
  static_assert(std::is_same<int, ns::element_t<1, ns::int2>>::value, "Error.");
}


void test_cpo()
{
  // test integral types
  {              int x{13}; assert(13 == ns::element<0>(x)); }
  {     unsigned int x{13}; assert(13 == ns::element<0>(x)); }
  {      std::size_t x{13}; assert(13 == ns::element<0>(x)); }
  { std::tuple<char> x{13}; assert(13 == ns::element<0>(x)); }

  // test homogeneous pairs
  {
    std::pair<int,int> x{13,7};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
  }

  // test heterogeneous 2-tuples
  {
    std::tuple<int, unsigned int> x{13, 7};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
  }

  // test 2-arrays
  {
    std::array<int, 2> x{13, 7};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
  }
  
  // test heterogeneous 3-tuples
  {
    std::tuple<int, unsigned int, std::size_t> x{13, 7, 42};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
    assert(42 == ns::element<2>(x));
  }

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

  // test heterogeneous pairs of homogeneous coordinates
  {
    std::pair<int2,uint3> x{{13,7}, {0,1,2}};
    assert((int2{13,7} == ns::element<0>(x)));
    assert((uint3{0,1,2} == ns::element<1>(x)));
  }

  // test heterogeneous tuples of homogeneous coordinates
  {
    std::tuple<int2,uint3,std::size_t> x{{13,7}, {0,1,2}, 42};
    assert((int2{13,7} == ns::element<0>(x)));
    assert((uint3{0,1,2} == ns::element<1>(x)));
    assert((size_t{42} == ns::element<2>(x)));
  }

  // test another level of tupling
  using uint2x3 = std::pair<uint3,uint3>;
  {
    uint2x3 x{{0,2,4}, {1,3,5}};
    assert((uint3{0,2,4} == ns::element<0>(x)));
    assert((uint3{1,3,5} == ns::element<1>(x)));
  }

  // test user-defined type with element member function
  {
    has_element_member_function x{13, 7, 42, 66};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
    assert(42 == ns::element<2>(x));
    assert(66 == ns::element<3>(x));
  }

  // test user-defined type with operator[]
  {
    has_operator_bracket x{13, 7, 42, 66};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
    assert(42 == ns::element<2>(x));
    assert(66 == ns::element<3>(x));
  }

  // test point
  {
    ns::point<int,3> x{13, 7, 42};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
    assert(42 == ns::element<2>(x));
  }

  // test typedefs
  {
    ns::int2 x{13, 7};
    assert(13 == ns::element<0>(x));
    assert( 7 == ns::element<1>(x));
  }
}


void test_element()
{
  test_cpo();
  test_element_t();
}

