#include <array>
#include <cassert>
#include <ubu/tensor/views/domain.hpp>
#include <ubu/tensor/views/layouts.hpp>
#include <ubu/tensor/views/slice.hpp>
#include <ubu/tensor/views/stack.hpp>
#include <utility>

// all elements of expected must exist
// negative elements of expected indicate that the
// corresponding element of result must not exist
template<ubu::tensor_like E, ubu::same_tensor_rank<E> R>
bool are_equal(E expected, R result)
{
  using namespace ubu;

  if(shape(expected) != shape(result)) return false;

  for(auto coord : domain(expected))
  {
    if(expected[coord] < 0)
    {
      if(element_exists(result, coord)) return false;
    }
    else
    {
      if(expected[coord] != result[coord]) return false;
    }
  }

  return true;
}

void test_slice()
{
  using namespace std;
  using namespace ubu;

  // XXX clang < version 17 gets CTAD wrong for these, so
  //     spell out the type of strided_layout
  strided_layout<ubu::int2, pair<constant<1>,int>> A(ubu::int2(3,2));
  strided_layout<ubu::int2, pair<constant<1>,int>> B(ubu::int2(2,2));

  // A is
  // { 0, 3,
  //   1, 4,
  //   2, 5 }
  //
  // B is
  // { 0, 2,
  //   1, 3 }

  {
    // test stack(A,B) on its own

    {
      // stack horizontally
      auto A_B = stack<1>(A,B);

      // A_B is
      // { 0, 3, 0, 2,
      //   1, 4, 1, 3,
      //   2, 5,-1,-1 };

      {
        // get column 1 of A_B
        std::array<int,3> expected = {3,4,5};
        auto result = slice(A_B, pair(_,1));

        // the result should have a particular type
        using expected_t = offset_layout<strided_layout<int,constant<1>,int>,int>;
        static_assert(std::same_as<expected_t, decltype(result)>);

        assert(are_equal(expected, result));
      }

      {
        // get column 2 of A_B
        // XXX it's a little strange that the result isn't {0,1,-1}
        //     we may want to fix stacked_view::slice to ensure that the returned
        //     slice has the expected shape
        std::array<int,2> expected = {0,1};
        auto result = slice(A_B, pair(_,2));

        // the result should have a particular type
        using expected_t = offset_layout<strided_layout<int,constant<1>,int>,int>;
        static_assert(std::same_as<expected_t, decltype(result)>);

        assert(are_equal(expected, result));
      }
    }
  }

  {
    // test compose(span, stack(A, B));

    std::array<int,12> array = { 0, 1,  2,  3,
                                 4, 5,  6,  7,
                                 8, 9, 10, 11 };
    std::span span(array.data(), array.size());

    // stack A & B horizontally, add an offset to B
    auto A_B = stack<1>(A,offset(B,A.size()));

    auto composition = compose(span, A_B);

    // composition is
    // { 0, 3,  6,  9,
    //   1, 4,  7, 10,
    //   2, 5, -1, -1 }
    
    {
      // get column 1 of composition
      std::array<int,3> expected = {3,4,5};
      auto result = slice(composition, pair(_,1));

      // the result should have a particular type
      using expected_t = composed_view<std::span<int>, strided_layout<int,constant<1>>>;
      static_assert(std::same_as<expected_t, decltype(result)>);

      assert(are_equal(expected, result));
    }

    {
      // get column 2 of composition
      // XXX it's a little strange that the result isn't {6,7,-1}
      //     we may want to fix stacked_view::slice to ensure that the returned
      //     slice has the expected shape
      std::array<int,2> expected = {6,7};
      auto result = slice(composition, pair(_,2));

      // the result should have a particular type
      using expected_t = composed_view<std::span<int>, strided_layout<int,constant<1>>>;
      static_assert(std::same_as<expected_t, decltype(result)>);

      assert(are_equal(expected, result));
    }
  }
}

