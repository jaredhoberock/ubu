#include <cassert>
#include <ubu/grid/coordinate/point.hpp>
#include <ubu/grid/slice/slice_coordinate.hpp>

namespace ns = ubu;

void test_slice_coordinate()
{
  using namespace std;
  using namespace ns;

  {
    int  coord    = 1;
    auto katana   = _;
    int  expected = 1;
    assert(expected == ns::slice_coordinate(coord, katana));
  }

  {
    ns::int2  coord(1,2);
    tuple    katana(_,2);
    int  expected = 1;
    assert(expected == ns::slice_coordinate(coord, katana));
  }

  {
    ns::int3   coord(1,2,3);
    tuple     katana(1,_,_);
    ns::int2  expected(2,3);
    assert(expected == ns::slice_coordinate(coord, katana));
  }

  {
    ns::int1    coord(6);
    auto     katana = _;
    ns::int1 expected(6);
    assert(expected == ns::slice_coordinate(coord, katana));
  }

  {
    tuple coord( ns::int3(1,2,3), ns::int1(6));
    tuple katana(ns::int3(1,2,3),          _);
    auto expected =               ns::int1(6);
    assert(expected == ns::slice_coordinate(coord, katana));
  }

  {
    tuple  coord(  ns::int3(1,2,3),   ns::int2(4,5), ns::int1(6));
    tuple katana(make_tuple(1,2,3), make_tuple(_,5), _);
    tuple expected(                            4,    ns::int1(6));
    assert(expected == ns::slice_coordinate(coord, katana));
  }

  {
    tuple    coord(tuple(tuple(1,2,3), tuple(4,5)));
    tuple   katana(tuple(tuple(_,2,_), _));
    tuple expected(            1,  3,  tuple(4,5));
    assert(expected == ns::slice_coordinate(coord, katana));
  }

  {
    tuple    coord(tuple(pair(1,2),3), tuple(4,5), tuple(6));
    tuple   katana(tuple(pair(_,2),_), _,          tuple(_));
    tuple expected(           1,   3,  tuple(4,5),       6);
    assert(expected == slice_coordinate(coord, katana));
  }
}
