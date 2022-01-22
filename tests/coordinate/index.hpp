#include <aspera/coordinate/index.hpp>
//#include <aspera/coordinate/point.hpp>
#include <utility>
#include <tuple>
#include <array>


namespace ns = aspera;


void test_index()
{
  // test some indices
  static_assert(ns::index<int>);
  static_assert(ns::index<unsigned int>);
  static_assert(ns::index<std::size_t>);
  static_assert(ns::index<std::tuple<char>>);
  //static_assert(ns::index<ns::point<int,1>>);
  static_assert(ns::are_indices<int, unsigned int, std::size_t, std::tuple<char>>);
  //static_assert(ns::are_indices<int, unsigned int, std::size_t, std::tuple<char>, ns::point<int,1>>);
  static_assert(ns::are_indices<int, unsigned int&, const std::size_t&, std::tuple<char&>>);
  //static_assert(ns::are_indices<int, unsigned int&, const std::size_t&, std::tuple<char&>, ns::point<int,1>&&>);

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

  static_assert(ns::index<int2>);
  //static_assert(ns::index<ns::int2>);
  static_assert(ns::index<uint3>);
  //static_assert(ns::index<ns::uint3>);
  static_assert(ns::index<std::tuple<int>>);
  static_assert(ns::index<std::tuple<int,unsigned int>>);
  static_assert(ns::index<std::tuple<int,unsigned int,std::size_t>>);
  static_assert(ns::index<std::array<std::size_t,3>>);

  static_assert(ns::index<std::pair<int2,uint3>>);
  static_assert(ns::index<std::tuple<int2,uint3,std::size_t>>);

  using uint2x3 = std::pair<uint3,uint3>;
  static_assert(ns::coordinate<uint2x3>);


  // test some types that are not discrete coordinates
  using float2 = std::pair<float,float>;
  using double3 = std::tuple<double, double, double>;

  static_assert(!ns::index<float>);
  static_assert(!ns::index<void>);
  static_assert(!ns::index<std::pair<float,int>>);
  static_assert(!ns::index<float2>);
  //static_assert(!ns::index<ns::float2>);
  static_assert(!ns::index<double3>);
  //static_assert(!ns::index<ns::double3>, "Error.");
  static_assert(!ns::index<int*>);
  static_assert(!ns::are_indices<int,float>);
  static_assert(!ns::are_indices<int&,float>);
}

