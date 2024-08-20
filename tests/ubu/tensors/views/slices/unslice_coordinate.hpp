#include <cassert>
#include <tuple>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/slices/slice_coordinate.hpp>
#include <ubu/tensors/views/slices/unslice_coordinate.hpp>

namespace ns = ubu;

void test_unslice_coordinate()
{
  using namespace std;
  using namespace ubu;

  {
    constexpr int  coord    = 1;
    constexpr auto katana   = _;

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr ns::int1 coord(1);
    constexpr auto katana = _;

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr ns::int2 coord(1,2);
    constexpr auto katana =  _;

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr ns::int2  coord(1,2);
    constexpr tuple    katana(_,2);

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr ns::int3 coord(1,2,3);
    constexpr tuple   katana(1,_,_);

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr ns::int1 coord(6);
    constexpr auto katana = _;

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr tuple  coord(ns::int3(1,2,3), ns::int1(6));
    constexpr tuple katana(ns::int3(1,2,3), _);

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr tuple  coord(  ns::int3(1,2,3),   ns::int2(4,5), ns::int1(6));
    constexpr tuple katana(make_tuple(1,2,3), make_tuple(_,5), _);

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr tuple    coord(tuple(tuple(1,2,3), tuple(4,5)));
    constexpr tuple   katana(tuple(tuple(_,2,_), _));

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr tuple    coord(tuple(pair(1,2),3), tuple(4,5), tuple(6));
    constexpr tuple   katana(tuple(pair(_,2),_), _,          tuple(_));

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr auto coord    = tuple(1, tuple());
    constexpr auto katana   = tuple(1, _);

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr int coord     = 1;
    constexpr auto katana   = 0;

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    // when katana contains no underscore, unslice_coordinate returns katana
    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(katana == unsliced_coord);
  }

  {
    constexpr ns::int2  coord(1,2);
    constexpr ns::int2 katana(3,4);

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    // when katana contains no underscore, unslice_coordinate returns katana
    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(katana == unsliced_coord);
  }

  {
    constexpr auto coord    = std::tuple();
    constexpr auto katana   = std::tuple();

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr auto coord    = tuple(1, tuple());
    constexpr auto katana   = tuple(1, _);
    
    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }

  {
    constexpr auto coord    = tuple(1, tuple(tuple(), tuple()));
    constexpr auto katana   = tuple(_, tuple(tuple(), _));

    constexpr auto sliced_coord = slice_coordinate(coord, katana);
    constexpr auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    static_assert(unslicer_for<decltype(katana), decltype(sliced_coord)>);
    static_assert(coord == unsliced_coord);
  }
}

