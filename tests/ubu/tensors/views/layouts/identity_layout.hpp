#include <cassert>
#include <ubu/tensors/concepts/tensor_like.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/concepts/view.hpp>
#include <ubu/tensors/iterators.hpp>
#include <ubu/tensors/views/lattice.hpp>
#include <ubu/tensors/views/layouts/concepts/layout.hpp>
#include <ubu/tensors/views/layouts/identity_layout.hpp>
#include <ubu/tensors/views/slices/slice.hpp>
#include <ranges>
#include <tuple>
#include <utility>

void test_identity_layout()
{
  using namespace ubu;

  {
    identity_layout l(10);

    using type = decltype(l);

    static_assert(tensor_like<type>);
    static_assert(view<type>);
    static_assert(layout<type>);
  }

  {
    identity_layout l(ubu::int2(2,3));

    using type = decltype(l);

    static_assert(tensor_like<type>);
    static_assert(view<type>);
    static_assert(layout<type>);

    {
      // test slice

      auto row_0 = slice(l, std::pair(0,_));
      std::vector<ubu::int2> expected_row_0{{ubu::int2(0,0), ubu::int2(0,1), ubu::int2(0,2)}};
      assert(std::ranges::equal(expected_row_0, row_0));

      auto row_1 = slice(l, std::pair(1,_));
      std::vector<ubu::int2> expected_row_1{{ubu::int2(1,0), ubu::int2(1,1), ubu::int2(1,2)}};
      assert(std::ranges::equal(expected_row_1, row_1));

      auto col_0 = slice(l, std::pair(_,0));
      std::vector<ubu::int2> expected_col_0{{ubu::int2(0,0), ubu::int2(1,0)}};
      assert(std::ranges::equal(expected_col_0, col_0));

      auto col_1 = slice(l, std::pair(_,1));
      std::vector<ubu::int2> expected_col_1{{ubu::int2(0,1), ubu::int2(1,1)}};
      assert(std::ranges::equal(expected_col_1, col_1));
    }
  }

  {
    identity_layout l(ubu::int4(2,3,4,5));

    using type = decltype(l);

    static_assert(tensor_like<type>);
    static_assert(view<type>);
    static_assert(layout<type>);

    {
      // test slice

      {
        auto s = slice(l, std::tuple(_,2,_,4));

        static_assert(tensor_like<decltype(s)>);
        static_assert(view<decltype(s)>);
        static_assert(layout<decltype(s)>);

        auto expected_s = lattice(ubu::int4(0,2,0,4), ubu::int4(2,1,4,1));

        assert(std::ranges::equal(expected_s, s));
      }
    }
  }

  {
    auto shape = std::tuple(std::pair(2,3), ubu::int3(4,5,6));

    identity_layout l(shape);

    using type = decltype(l);

    static_assert(tensor_like<type>);
    static_assert(view<type>);
    static_assert(layout<type>);
  }
}
