#include <algorithm>
#include <cassert>
#include <numeric>
#include <ranges>
#include <ubu/tensors/coordinates/comparisons.hpp>
#include <ubu/tensors/iterators.hpp>
#include <ubu/tensors/matrices/column_major_layout.hpp>
#include <ubu/tensors/shapes/shape_size.hpp>
#include <ubu/tensors/views/composed_view.hpp>
#include <ubu/tensors/views/domain.hpp>
#include <ubu/tensors/views/lattice.hpp>
#include <ubu/tensors/views/layouts/compact_left_major_layout.hpp>
#include <ubu/tensors/views/layouts/compact_right_major_layout.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace ns = ubu;

template<ns::coordinate S>
void test(S shape)
{
  using namespace ns;

  {
    // left-major view of a lattice

    lattice tensor(shape);
    compact_left_major_layout layout(shape);

    composed_view v(tensor, layout);

    for(auto c : domain(v))
    {
      auto result = v[c];
      auto expected = tensor[c];
      assert(expected == result);
    }

    assert(std::ranges::is_sorted(domain(v), ns::colex_less));
  }

  {
    // right-major view of a lattice

    lattice tensor(shape);
    compact_right_major_layout layout(shape);

    composed_view v(tensor, layout);

    for(auto c : domain(v))
    {
      auto result = v[c];
      auto expected = tensor[ns::apply_stride(ns::compact_right_major_stride(shape), c)];
      assert(expected == result);
    }
  }
}

void test_slice()
{
  using namespace ns;
  using namespace std;

  ns::int2 shape(3,3);
  vector<int> tensor(shape_size(shape));
  iota(tensor.begin(), tensor.end(), 0);

  {
    // test slice of a composed_view<T*,...>
    composed_view A(tensor.data(), column_major_layout(shape));

    // A is
    // +---+---+---+
    // | 0 | 3 | 6 |
    // +---+---+---+
    // | 1 | 4 | 7 |
    // +---+---+---+
    // | 2 | 5 | 8 |
    // +---+---+---+

    array<int,3> expected{{3,4,5}};

    auto column_1 = A.slice(pair(_,1));

    static_assert(std::same_as<decltype(column_1), composed_view<int*, strided_layout<int, constant<1>>>>);

    assert(ranges::equal(expected, column_1));
  }

  {
    // test slice of a composed_view<std::span<T>, ...>
    composed_view A(span(tensor.data(), tensor.size()), column_major_layout(shape));

    // A is
    // +---+---+---+
    // | 0 | 3 | 6 |
    // +---+---+---+
    // | 1 | 4 | 7 |
    // +---+---+---+
    // | 2 | 5 | 8 |
    // +---+---+---+

    array<int,3> expected{{3,4,5}};

    auto column_1 = A.slice(pair(_,1));

    static_assert(std::same_as<decltype(column_1), composed_view<span<int>, strided_layout<int, constant<1>>>>);

    assert(ranges::equal(expected, column_1));
  }
}

void test_composed_view()
{
  using namespace std;

  size_t n = 12345;

  test(n);
  test(ns::convert_shape<ns::int2>(n));
  test(ns::convert_shape<ns::int3>(n));
  test(ns::convert_shape<pair<int, ns::int2>>(n));
  test(ns::convert_shape<pair<ns::int3, ns::int2>>(n));

  test_slice();
}

