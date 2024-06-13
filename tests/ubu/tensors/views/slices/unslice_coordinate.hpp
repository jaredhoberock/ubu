#include <cassert>
#include <tuple>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/slices/slice_coordinate.hpp>
#include <ubu/tensors/views/slices/unslice_coordinate.hpp>

namespace ns = ubu;

void test_unslice_coordinate()
{
  using namespace std;
  using namespace ns;

  {
    int  coord    = 1;
    auto katana   = _;

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    ns::int1 coord(1);
    auto katana = _;

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    ns::int2 coord(1,2);
    auto katana =  _;

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    ns::int2 coord(1,2);
    tuple   katana(_,2);

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    ns::int3 coord(1,2,3);
    tuple   katana(1,_,_);

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    ns::int1 coord(6);
    auto  katana = _;

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    tuple  coord(ns::int3(1,2,3), ns::int1(6));
    tuple katana(ns::int3(1,2,3), _);

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    tuple  coord(  ns::int3(1,2,3),   ns::int2(4,5), ns::int1(6));
    tuple katana(make_tuple(1,2,3), make_tuple(_,5), _);

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    tuple    coord(tuple(tuple(1,2,3), tuple(4,5)));
    tuple   katana(tuple(tuple(_,2,_), _));

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }

  {
    tuple    coord(tuple(pair(1,2),3), tuple(4,5), tuple(6));
    tuple   katana(tuple(pair(_,2),_), _,          tuple(_));

    auto sliced_coord = slice_coordinate(coord, katana);
    auto unsliced_coord = unslice_coordinate(sliced_coord, katana);

    assert(coord == unsliced_coord);
  }
}

