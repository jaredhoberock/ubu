#include <array>
#include <cassert>
#include <ubu/tensor/domain.hpp>
#include <ubu/tensor/stack.hpp>
#include <ubu/tensor/layout.hpp>
#include <ubu/tensor/view.hpp>

namespace ns = ubu;

// all elements of expected must exist
// negative elements of expected indicate that the
// corresponding element of result must not exist
template<ubu::matrix_like E, ubu::matrix_like R>
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

void test_stack()
{
  using namespace ns;

  {
    strided_layout A(ubu::int2(3,2));
    strided_layout B(ubu::int2(2,2));

    {
      // stack horizontally
      auto result = stack<1>(A,B);

      std::array<int,12> expected_data = { 0, 3, 0, 2,
                                           1, 4, 1, 3,
                                           2, 5,-1,-1 };
      view expected(expected_data.data(), row_major(ns::int2(3,4)));

      assert(are_equal(expected, result));
    }

    {
      // stack horizontally in the opposite direction
      auto result = stack<1>(B,A);

      std::array<int,12> expected_data = { 0, 2, 0, 3,
                                           1, 3, 1, 4,
                                          -1,-1, 2, 5 };
      view expected(expected_data.data(), row_major(ns::int2(3,4)));

      assert(are_equal(expected, result));
    }
  }

  {
    strided_layout A(ubu::int2(3,2));
    strided_layout B(ubu::int2(2,2));

    {
      // stack vertically
      auto result = stack<0>(A,B);

      std::array<int,10> expected_data = { 0, 3,
                                           1, 4,
                                           2, 5,
                                           0, 2,
                                           1, 3 };

      view expected(expected_data.data(), row_major(ns::int2(5,2)));

      assert(are_equal(expected, result));
    }

    {
      // stack vertically in the opposite direction
      auto result = stack<0>(B,A);

      std::array<int,10> expected_data = { 0, 2,
                                           1, 3,
                                           0, 3,
                                           1, 4,
                                           2, 5 };

      view expected(expected_data.data(), row_major(ns::int2(5,2)));

      assert(are_equal(expected, result));
    }
  }

  {
    strided_layout A(3);
    strided_layout B(2);

    {
      // stack horizontally
      auto result = stack<1>(A,B);

      std::array<int,6> expected_data = { 0, 0,
                                          1, 1,
                                          2,-1 };
      view expected(expected_data.data(), row_major(ns::int2(3,2)));

      assert(are_equal(expected, result));
    }

    {
      // stack horizontally in the opposite direction
      auto result = stack<1>(B,A);

      std::array<int,6> expected_data = { 0, 0,
                                          1, 1,
                                         -1, 2 };
      view expected(expected_data.data(), row_major(ns::int2(3,2)));

      assert(are_equal(expected, result));
    }
  }
}

