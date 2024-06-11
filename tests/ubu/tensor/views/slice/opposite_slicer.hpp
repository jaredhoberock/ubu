#include <cassert>
#include <ubu/tensor/coordinates/point.hpp>
#include <ubu/tensor/views/slice/dice_coordinate.hpp>

namespace ns = ubu;

void test_opposite_slicer()
{
  using namespace std;
  using namespace ns;

  {
    constexpr int  coord    = 1;
    constexpr auto katana   = _;
    constexpr int  expected = 1;
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr ns::int2  coord(1,2);
    constexpr tuple    katana(_,2);
    constexpr tuple  expected(1,_);
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr ns::int3   coord(1,2,3);
    constexpr tuple     katana(1,_,_);
    constexpr tuple   expected(_,2,3);
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr ns::int1    coord(6);
    constexpr auto     katana = _;
    constexpr ns::int1 expected(6);
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr tuple    coord( ns::int3(1,2,3), ns::int1(6));
    constexpr tuple    katana(ns::int3(1,2,3),          _);
    constexpr tuple  expected(              _, ns::int1(6));
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr tuple  coord(    ns::int3(1,2,3),   ns::int2(4,5), ns::int1(6));
    constexpr tuple katana(  make_tuple(1,2,3), make_tuple(_,5), _);
    constexpr tuple expected(                _, make_tuple(4,_), ns::int1(6));
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr tuple    coord(tuple(tuple(1,2,3), tuple(4,5)));
    constexpr tuple   katana(tuple(tuple(_,2,_), _));
    constexpr tuple expected(tuple(tuple(1,_,3), tuple(4,5)));
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr tuple    coord(tuple(pair(1,2),3), tuple(4,5), tuple(6));
    constexpr tuple   katana(tuple(pair(_,2),_), _,          tuple(_));
    constexpr tuple expected(tuple(pair(1,_),3), tuple(4,5), tuple(6));
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }

  {
    constexpr tuple    coord(1,_,_);
    constexpr tuple   katana(1,_,_);
    constexpr tuple expected(_,0,0);
    static_assert(expected == ns::opposite_slicer(coord, katana));
  }
}

