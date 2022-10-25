#include <ubu/coordinate/weakly_congruent.hpp>
#include <array>
#include <tuple>
#include <utility>

void test_weakly_congruent()
{
  using namespace ubu;

  // test some weakly congruent scalars
  static_assert(weakly_congruent<int,int>);
  static_assert(weakly_congruent<int,unsigned int>);
  static_assert(weakly_congruent<unsigned int,int>);
  static_assert(weakly_congruent<float,float>);
  static_assert(weakly_congruent<double,double>);
  static_assert(weakly_congruent<double,float>);
  static_assert(weakly_congruent<float,double>);
  static_assert(weakly_congruent<char, int>);

  // test some weakly congruent coordinates of different rank
  static_assert(weakly_congruent<int, std::pair<int,int>>);
  static_assert(weakly_congruent<int, std::tuple<int,int,int>>);
  static_assert(weakly_congruent<std::pair<int,int>, std::pair<int,std::pair<int,int>>>);

  // test some weakly congruent references
  static_assert(weakly_congruent<unsigned int&, std::tuple<int&&, const volatile unsigned int &, char>&&>);
  
  // test some tuple_like coordinates
  static_assert(weakly_congruent<int, std::tuple<int>>);
  static_assert(weakly_congruent<std::tuple<int>, std::tuple<char>>);
  static_assert(weakly_congruent<std::tuple<int>, std::tuple<int,char>>);
  static_assert(weakly_congruent<std::tuple<int,std::size_t>, std::pair<char, int>>);
  static_assert(weakly_congruent<std::array<std::size_t,3>, std::tuple<int,int,int>>);

  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;
  using uint2x3 = std::pair<uint3,uint3>;
  using int2x3 = std::tuple<std::array<int,3>,std::array<int,3>>;
  static_assert(weakly_congruent<uint2x3, int2x3>);


  // test some types that are not weakly congruent
  using float2 = std::pair<float,float>;
  using double3 = std::tuple<double, double, double>;

  static_assert(!weakly_congruent<char,double>);
  static_assert(!weakly_congruent<float,int>);
  static_assert(!weakly_congruent<int*,int>);
  static_assert(!weakly_congruent<int*,int*>);
  static_assert(!weakly_congruent<float2,double3>);
  static_assert(!weakly_congruent<uint3, uint2x3>);
  static_assert(!weakly_congruent<std::array<int,1>, int>);
  static_assert(!weakly_congruent<int&, float>);
}

