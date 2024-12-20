#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/slices/slice_coordinate.hpp>

namespace ns = ubu;

void test_slice_coordinate()
{
  using namespace std;
  using namespace ns;

  {
    constexpr int  coord    = 1;
    constexpr auto katana   = _;
    constexpr int  expected = 1;
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr ns::int2  coord(1,2);
    constexpr tuple    katana(_,2);
    constexpr int  expected = 1;
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr ns::int3   coord(1,2,3);
    constexpr tuple     katana(1,_,_);
    constexpr ns::int2  expected(2,3);
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr ns::int1    coord(6);
    constexpr auto     katana = _;
    constexpr ns::int1 expected(6);
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr tuple coord( ns::int3(1,2,3), ns::int1(6));
    constexpr tuple katana(ns::int3(1,2,3), _);
    constexpr auto expected =               ns::int1(6);
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr tuple  coord(  ns::int3(1,2,3),     ns::int2(4,5), ns::int1(6));
    constexpr tuple katana(make_tuple(1,2,3),   make_tuple(_,5), _);
    constexpr tuple expected(                              4,    ns::int1(6));
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr tuple    coord(tuple(tuple(1,2,3), tuple(4,5)));
    constexpr tuple   katana(tuple(tuple(_,2,_), _));
    constexpr tuple expected(            1,  3,  tuple(4,5));
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr tuple    coord(tuple(pair(1,2),3), tuple(4,5), tuple(6));
    constexpr tuple   katana(tuple(pair(_,2),_), _,          tuple(_));
    constexpr tuple expected(           1,   3,  tuple(4,5),       6);
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr int coord     = 1;
    constexpr auto katana   = 0;
    constexpr auto expected = std::tuple();
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr ubu::int2  coord(1,2);
    constexpr ubu::int2 katana(0,0);
    constexpr auto expected = std::tuple();
    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr auto coord    = std::tuple();
    constexpr auto katana   = std::tuple();
    constexpr auto expected = std::tuple();

    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr auto coord    = std::tuple(1, std::tuple());
    constexpr auto katana   = std::tuple(0, _);
    constexpr auto expected = std::tuple();

    static_assert(expected == slice_coordinate(coord, katana));
  }

  {
    constexpr auto coord    = std::tuple(1, std::tuple(std::tuple(), std::tuple()));
    constexpr auto katana   = std::tuple(_, std::tuple(std::tuple(), _));
    constexpr auto expected = std::tuple(1,                          std::tuple());

    static_assert(expected == slice_coordinate(coord, katana));
  }
}

