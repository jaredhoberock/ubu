#include <cassert>
#include <ubu/tensors/concepts/view.hpp>
#include <ubu/tensors/concepts/tensor_like_of_rank.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/iterators.hpp>
#include <ubu/tensors/views/lattice.hpp>
#include <ubu/tensors/views/layouts/concepts/layout.hpp>
#include <ubu/tensors/views/layouts/quilting_layout.hpp>
#include <ubu/tensors/views/slices.hpp>
#include <ranges>
#include <tuple>
#include <utility>

void test_quilting_layout()
{
  using namespace ubu;

  {
    // concatenate (1D, 1D)

    int inner_shape = 10;
    int outer_shape = 5;
    std::tuple expected_shape(inner_shape, outer_shape);

    quilting_layout l(inner_shape, outer_shape);

    assert(expected_shape == l.shape());

    static_assert(layout<decltype(l)>);
    static_assert(view<decltype(l)>);

    for(int i = 0; i < outer_shape; ++i)
    {
      for(int j = 0; j < inner_shape; ++j)
      {
        assert(std::pair(i,j) == l[std::pair(i,j)]);
      }
    }

    auto s = slice(l, _);

    static_assert(std::same_as<decltype(s), decltype(l)>);
    assert(std::ranges::equal(s, l));
  }

  {
    // concatenate (2D, 2D)

    ubu::int2 inner_shape(2,3);
    ubu::int2 outer_shape(4,5);
    ubu::int4 expected_shape(2,3,4,5);

    quilting_layout l(inner_shape, outer_shape);

    assert(expected_shape == l.shape()); 

    static_assert(layout<decltype(l)>);
    static_assert(view<decltype(l)>);

    for(ubu::int2 outer_coord : lattice(outer_shape))
    {
      for(ubu::int2 inner_coord : lattice(inner_shape))
      {
        ubu::int4 coord = coordinate_cat(inner_coord, outer_coord);

        std::pair expected(inner_coord, outer_coord);

        assert(expected == l[coord]);
      }
    }

    {
      // test slice

      {
        // a 1D slice

        auto s = slice(l, std::tuple(_,1,2,3));

        static_assert(tensor_like_of_rank<decltype(s),1>);
        assert(2 == shape(s));

        for(int i : lattice(2))
        {
          std::pair expected(ubu::int2(i,1), ubu::int2(2,3));
          assert(expected == s[i]);
        }
      }

      {
        // a 2D slice

        auto s = slice(l, std::tuple(_,1,2,_));

        static_assert(tensor_like_of_rank<decltype(s),2>);
        assert(ubu::int2(2,5) == shape(s));

        using expected_slice_type = quilting_layout<
          sliced_view<identity_layout<ubu::int2>, std::tuple<ubu::detail::underscore_t,int>>,
          sliced_view<identity_layout<ubu::int2>, std::tuple<int,ubu::detail::underscore_t>>
        >;

        static_assert(std::same_as<expected_slice_type, decltype(s)>);

        for(int i : lattice(2))
        {
          for(int j : lattice(5))
          {
            std::pair expected(ubu::int2(j,1), ubu::int2(2,i));

            ubu::int2 coord(j,i);

            assert(expected == s[coord]);
          }
        }
      }
    }
  }
}

