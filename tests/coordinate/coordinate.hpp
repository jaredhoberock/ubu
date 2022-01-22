#include <array>
#include <aspera/coordinate/coordinate.hpp>
#include <tuple>
#include <utility>


namespace ns = aspera;


struct has_member_functions
{
  static constexpr std::size_t size()
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


void test_coordinate()
{
  // test a user-defined coordinate
  static_assert(ns::coordinate<has_member_functions>);

  // test some integer coordinates
  static_assert(ns::coordinate<int>);
  static_assert(ns::coordinate<int&>);
  static_assert(ns::coordinate<unsigned int>);
  static_assert(ns::coordinate<const unsigned int&>);
  static_assert(ns::coordinate<std::size_t>);
  static_assert(ns::coordinate<volatile std::size_t&>);
  static_assert(ns::coordinate<std::tuple<char>>);
  static_assert(ns::coordinate<std::tuple<char&>>);
  static_assert(ns::coordinate<std::tuple<int, char&>>);

  // test some floating point coordinates
  static_assert(ns::coordinate<float>);
  static_assert(ns::coordinate<double>);
  static_assert(ns::coordinate<std::tuple<float>>);
  static_assert(ns::coordinate<std::tuple<double>>);
  static_assert(ns::coordinate<std::tuple<float&,float>>);

  // test some multidimensional coordinates
  static_assert(ns::coordinate<std::pair<int,int>>);
  static_assert(ns::coordinate<std::tuple<int>>);
  static_assert(ns::coordinate<std::tuple<int,unsigned int>>);
  static_assert(ns::coordinate<std::tuple<int,unsigned int,std::size_t>>);
  static_assert(ns::coordinate<std::array<std::size_t,3>>);
  //static_assert(ns::coordinate<ns::float2>);
  //static_assert(ns::coordinate<ns::uint3>);

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

  static_assert(ns::coordinate<std::pair<int2,uint3>>);
  static_assert(ns::coordinate<std::tuple<int2,uint3,std::size_t>>);

  using uint2x3 = std::pair<uint3,uint3>;
  static_assert(ns::coordinate<uint2x3>);

  // test some non-coordinates
  static_assert(!ns::coordinate<std::tuple<int, char*>>);
}

