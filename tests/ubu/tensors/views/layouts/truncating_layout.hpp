#include <cassert>
#include <ubu/tensors/coordinates/comparisons/coordinate_equal.hpp>
#include <ubu/tensors/coordinates/one_extend_coordinate.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/coordinates/truncate_coordinate.hpp>
#include <ubu/tensors/views/lattice.hpp>
#include <ubu/tensors/views/layouts/truncating_layout.hpp>
#include <ubu/utilities/tuples.hpp>
#include <tuple>

template<ubu::coordinate From, ubu::coordinate To>
  requires ubu::subdimensional<To,From>
void test(To to)
{
  using namespace ubu;

  auto from = one_extend_coordinate<From>(to);

  truncating_layout<From,To> layout(to);

  assert(coordinate_equal(from, layout.shape()));
  assert(coordinate_equal(to, layout.coshape()));

  for(auto coord : lattice(from))
  {
    auto expected = truncate_coordinate<To>(coord);

    assert(coordinate_equal(expected, layout[coord]));
  }
}

void test_truncating_layout()
{
  {
    test<ubu::int2>(2);
    test<ubu::int2>(std::tuple(2));
  }

  {
    test<ubu::int3>(2);
    test<ubu::int3>(std::tuple(2));
    test<ubu::int3>(std::tuple(2,3));
  }

  {
    test<ubu::int4>(2);
    test<ubu::int4>(std::tuple(2));
    test<ubu::int4>(std::tuple(2,3));
    test<ubu::int4>(std::tuple(2,3,4));
  }

  {
    test<ubu::int5>(2);
    test<ubu::int5>(std::tuple(2));
    test<ubu::int5>(std::tuple(2));
    test<ubu::int5>(std::tuple(2,3));
    test<ubu::int5>(std::tuple(2,3,4));
    test<ubu::int5>(std::tuple(2,3,4,5));
  }

  {
    std::tuple from(2, std::tuple(3,4), std::tuple(5,6));

    using From = decltype(from);

    test<From>(2);
    test<From>(ubu::tuples::take<1>(from));
    test<From>(ubu::tuples::take<2>(from));
  }
}

