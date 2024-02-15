#include <algorithm>
#include <cassert>
#include <numeric>
#include <ubu/tensor/coordinate/compare.hpp>
#include <ubu/tensor/domain.hpp>
#include <ubu/tensor/lattice.hpp>
#include <ubu/tensor/layout/column_major.hpp>
#include <ubu/tensor/layout/row_major.hpp>
#include <ubu/tensor/shape/shape_size.hpp>
#include <ubu/tensor/view.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace ns = ubu;

template<ns::coordinate S>
void test(S shape)
{
  using namespace ns;

  {
    // column major view of a lattice

    lattice tensor(shape);
    column_major layout(shape);

    view v(tensor, layout);

    for(auto c : domain(v))
    {
      auto result = v[c];
      auto expected = tensor[c];
      assert(expected == result);
    }

    assert(std::is_sorted(domain(v).begin(), domain(v).end(), ns::colex_less));
  }

  {
    // row major view of a lattice

    lattice tensor(shape);
    row_major layout(shape);

    view v(tensor, layout);

    for(auto c : domain(v))
    {
      auto result = v[c];
      auto expected = tensor[ns::apply_stride(ns::compact_row_major_stride(shape), c)];
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
    // test slice of a view<T*,...>
    view A(tensor.data(), column_major(shape));

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

    static_assert(std::same_as<decltype(column_1), view<int*, strided_layout<int, constant<1>>>>);

    assert(equal(expected.begin(), expected.end(), column_1.begin()));
  }

  {
    // test slice of a view<std::span<T>, ...>
    view A(span(tensor.data(), tensor.size()), column_major(shape));

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

    static_assert(std::same_as<decltype(column_1), view<span<int>, strided_layout<int, constant<1>>>>);

    assert(equal(expected.begin(), expected.end(), column_1.begin()));
  }
}

void test_view()
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

