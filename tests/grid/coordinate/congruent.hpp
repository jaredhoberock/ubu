#include <ubu/grid/coordinate/congruent.hpp>
#include <array>
#include <tuple>
#include <utility>

void test_congruent()
{
  using namespace ubu;

  // test some congruent coordinates
  static_assert(congruent<int,int>);
  static_assert(congruent<int,unsigned int>);
  static_assert(congruent<unsigned int,int>);
  static_assert(congruent<char, int, unsigned int, char>);
  static_assert(congruent<int, std::array<int,1>>);

  // test some congruent references
  static_assert(congruent<unsigned int&, int&&, const volatile unsigned int &, char>);
  
  static_assert(congruent<std::tuple<int>, std::tuple<int>>);
  static_assert(congruent<std::tuple<int,std::size_t>, std::pair<char, int>>);
  static_assert(congruent<std::array<std::size_t,3>, std::tuple<int,int,int>>);

  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;
  using uint2x3 = std::pair<uint3,uint3>;
  using int2x3 = std::tuple<std::array<int,3>,std::array<int,3>>;
  static_assert(congruent<uint2x3, int2x3>);


  // test some types that are not congruent
  static_assert(!congruent<char,double>);
  static_assert(!congruent<float,int>);
  static_assert(!congruent<int*,int>);
  static_assert(!congruent<int*,int*>);
  static_assert(!congruent<int,uint2x3>);
  static_assert(!congruent<uint3, uint2x3>);
  static_assert(!congruent<std::array<int,2>, int>);
  static_assert(!congruent<int, unsigned int, char, float>);
}

