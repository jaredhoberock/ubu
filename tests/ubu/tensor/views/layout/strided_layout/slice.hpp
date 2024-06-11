#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <ranges>
#include <ubu/tensors/iterators.hpp>
#include <ubu/tensors/views/layouts/strided_layout.hpp>
#include <ubu/tensors/views/layouts/stride/compact_row_major_stride.hpp>
#include <utility>

namespace ns = ubu;

void test_slice()
{
  using namespace ns;
  using namespace std;

  {
    strided_layout A(ns::int2(3,4));

    // A is
    // +---+---+---+----+
    // | 0 | 3 | 6 |  9 |
    // +---+---+---+----+
    // | 1 | 4 | 7 | 10 |
    // +---+---+---+----+
    // | 2 | 5 | 8 | 11 |
    // +---+---+---+----+

    auto column_2 = A.slice(pair(_,2));

    std::array<int,3> expected{{6,7,8}};

    assert(ranges::equal(expected, column_2));
  }

  {
    ns::int2 shape(3,4);
    strided_layout A(shape, compact_row_major_stride(shape));

    // A is
    // +---+---+----+----+
    // | 0 | 1 |  2 |  3 |
    // +---+---+----+----+
    // | 4 | 5 |  6 |  7 |
    // +---+---+----+----+
    // | 8 | 9 | 10 | 11 |
    // +---+---+----+----+

    auto column_2 = A.slice(pair(_,2));

    std::array<int,3> expected{{2,6,10}};

    assert(ranges::equal(expected, column_2));
  }
}

