#include <aspera/coordinate/discrete_coordinate.hpp>
#include <aspera/coordinate/point.hpp>
#include <utility>
#include <tuple>
#include <array>


namespace ns = aspera;


void test_discrete_coordinate()
{
  // test some discrete coordinates
  static_assert(ns::discrete_coordinate<int>);
  static_assert(ns::discrete_coordinate<unsigned int>);
  static_assert(ns::discrete_coordinate<std::size_t>);
  static_assert(ns::discrete_coordinate<std::tuple<char>>);
  static_assert(ns::discrete_coordinate<ns::point<int,1>>);
  static_assert(ns::are_discrete_coordinates<int, unsigned int, std::size_t, std::tuple<char>>);
  static_assert(ns::are_discrete_coordinates<int, unsigned int, std::size_t, std::tuple<char>, ns::point<int,1>>);
  static_assert(ns::are_discrete_coordinates<int, unsigned int&, const std::size_t&, std::tuple<char&>>);
  static_assert(ns::are_discrete_coordinates<int, unsigned int&, const std::size_t&, std::tuple<char&>, ns::point<int,1>&&>);

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

  static_assert(ns::discrete_coordinate<int2>);
  static_assert(ns::discrete_coordinate<ns::int2>);
  static_assert(ns::discrete_coordinate<uint3>);
  static_assert(ns::discrete_coordinate<ns::uint3>);
  static_assert(ns::discrete_coordinate<std::tuple<int>>);
  static_assert(ns::discrete_coordinate<std::tuple<int,unsigned int>>);
  static_assert(ns::discrete_coordinate<std::tuple<int,unsigned int,std::size_t>>);
  static_assert(ns::discrete_coordinate<std::array<std::size_t,3>>);

  static_assert(ns::discrete_coordinate<std::pair<int2,uint3>>);
  static_assert(ns::discrete_coordinate<std::tuple<int2,uint3,std::size_t>>);

  using uint2x3 = std::pair<uint3,uint3>;
  static_assert(ns::coordinate<uint2x3>);


  // test some types that are not discrete coordinates
  using float2 = std::pair<float,float>;
  using double3 = std::tuple<double, double, double>;

  static_assert(!ns::discrete_coordinate<float>);
  static_assert(!ns::discrete_coordinate<void>);
  static_assert(!ns::discrete_coordinate<std::pair<float,int>>);
  static_assert(!ns::discrete_coordinate<float2>);
  static_assert(!ns::discrete_coordinate<ns::float2>);
  static_assert(!ns::discrete_coordinate<double3>);
  static_assert(!ns::discrete_coordinate<ns::double3>, "Error.");
  static_assert(!ns::discrete_coordinate<int*>);
  static_assert(!ns::are_discrete_coordinates<int,float>);
  static_assert(!ns::are_discrete_coordinates<int&,float>);
}

