#include <cassert>
#include <ubu/tensors/coordinates/comparisons/coordinate_equal.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/coordinates/one_extend_coordinate.hpp>
#include <ubu/tensors/coordinates/zero_extend_coordinate.hpp>
#include <ubu/tensors/views/lattice.hpp>
#include <ubu/tensors/views/layouts/extending_layout.hpp>
#include <ubu/utilities/tuples.hpp>
#include <tuple>

template<ubu::coordinate To, ubu::coordinate From>
  requires ubu::superdimensional<To,From>
void test(From from)
{
  using namespace ubu;

  auto to = one_extend_coordinate<To>(from);

  extending_layout<From,To> layout(from);

  assert(coordinate_equal(from, layout.shape()));
  assert(coordinate_equal(to, layout.coshape()));

  for(auto coord : lattice(from))
  {
    auto expected = zero_extend_coordinate<To>(coord);

    assert(coordinate_equal(expected, layout[coord]));
  }
}

void test_extending_layout()
{
  {
    test<ubu::int2>(2);
    test<ubu::int2>(std::tuple(2));
    test<ubu::int2>(std::tuple(2, 3));
  }

  {
    test<ubu::int3>(2);
    test<ubu::int3>(std::tuple(2,3));
    test<ubu::int3>(std::tuple(2,3,4));
  }

  {
    test<ubu::int4>(2);
    test<ubu::int4>(std::tuple(2,3));
    test<ubu::int4>(std::tuple(2,3,4));
    test<ubu::int4>(std::tuple(2,3,4,5));
  }

  {
    test<ubu::int5>(2);
    test<ubu::int5>(std::tuple(2,3));
    test<ubu::int5>(std::tuple(2,3,4));
    test<ubu::int5>(std::tuple(2,3,4,5));
    test<ubu::int5>(std::tuple(2,3,4,5,6));
  }

  {
    std::tuple to(2, std::tuple(3,4), std::tuple(5,6));

    using To = decltype(to);

    test<To>(2);
    test<To>(ubu::tuples::take<1>(to));
    test<To>(ubu::tuples::take<2>(to));
  }
}

