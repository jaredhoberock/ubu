#include <ubu/miscellaneous/constant.hpp>
#include <ubu/miscellaneous/integrals/bounded.hpp>
#include <functional>
#include <iostream>

template<int a, int b, class Op>
constexpr void test_operation_with_normal_integer(Op op)
{
  using namespace ubu;

  // bounded op int
  {
    constexpr bounded<a> lhs = a;
    constexpr int rhs = b;
    constexpr auto result = op(lhs, rhs);

    if constexpr(std::same_as<Op,std::divides<>>)
    {
      // bounded / int -> bounded is an exception because it yields a bounded instead of an int
      constexpr bounded<a> expected = op(a, b);
    
      static_assert(expected == result);
      static_assert(std::same_as<decltype(expected), decltype(result)>);
    }
    else
    {
      constexpr int expected = op(a, b);
  
      static_assert(expected == result);
      static_assert(std::same_as<decltype(expected), decltype(result)>);
    }
  }

  // int op bounded
  {
    constexpr int lhs = a;
    constexpr bounded<b> rhs = b;
    constexpr auto result = op(lhs, rhs);

    if constexpr(std::same_as<Op,std::modulus<>>)
    {
      // int % bounded -> bounded is an exception because it yields a bounded instead of an int
      constexpr bounded<b-1> expected = op(a, b);

      static_assert(expected == result);
      static_assert(std::same_as<decltype(expected), decltype(result)>);
    }
    else
    {
      constexpr int expected = op(a, b);

      static_assert(expected == result);
      static_assert(std::same_as<decltype(expected), decltype(result)>);
    }
  }
}

void test_bounded()
{
  using namespace ubu;

  // arithmetic with normal integer
  {
    test_operation_with_normal_integer<3,4>(std::plus());
    test_operation_with_normal_integer<3,4>(std::minus());
    test_operation_with_normal_integer<3,4>(std::multiplies());
    test_operation_with_normal_integer<3,4>(std::divides());
    test_operation_with_normal_integer<3,4>(std::modulus());
  }

  // arithmetic with bounded

  // bounded + bounded -> bounded
  {
    constexpr bounded<10> lhs = 3;
    constexpr bounded<20> rhs = 4;
    constexpr bounded<30> expected = 7;
    constexpr auto result = lhs + rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded - bounded -> bounded
  {
    constexpr bounded<10> lhs = 4;
    constexpr bounded<5>  rhs = 3;
    constexpr bounded<10> expected = 4 - 3;
    constexpr auto result = lhs - rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded - bounded -> int
  {
    constexpr bounded<10> lhs = 4;
    constexpr bounded<20> rhs = 3;
    constexpr int expected = 4 - 3;
    constexpr auto result = lhs - rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded * bounded -> bounded
  {
    constexpr bounded<10> lhs = 3;
    constexpr bounded<20> rhs = 4;
    constexpr bounded<200> expected = 3 * 4;
    constexpr auto result = lhs * rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded / bounded -> bounded
  {
    constexpr bounded<10> lhs = 8;
    constexpr bounded<20> rhs = 2;
    constexpr bounded<10> expected = 8 / 2;
    constexpr auto result = lhs / rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded % bounded -> bounded
  {
    constexpr bounded<10> lhs = 8;
    constexpr bounded<20> rhs = 2;
    constexpr bounded<19> expected = 8 % 2;
    constexpr auto result = lhs % rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // arithmetic with constant

  // bounded + constant -> bounded
  {
    constexpr bounded<10> lhs = 7;
    constexpr auto rhs = 5_c;
    constexpr bounded<15> expected = 7 + 5;
    constexpr auto result = lhs + rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // constant + bounded -> bounded
  {
    constexpr auto lhs = 4_c;
    constexpr bounded<10> rhs = 3;
    constexpr bounded<14> expected = 4 + 3;
    constexpr auto result = lhs + rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded - constant -> bounded
  {
    constexpr bounded<10> lhs = 7;
    constexpr auto rhs = 5_c;
    constexpr bounded<5> expected = 7 - 5;
    constexpr auto result = lhs - rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded - constant -> int
  {
    constexpr bounded<10> lhs = 7;
    constexpr auto rhs = 9_c;
    constexpr bounded<1> expected = 7 - 9;
    constexpr auto result = lhs - rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // constant - bounded -> int
  {
    constexpr auto lhs = 7_c;
    constexpr bounded<10> rhs = 9;
    constexpr int expected = 7 - 9;
    constexpr auto result = lhs - rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded * constant -> bounded
  {
    constexpr bounded<4> lhs = 3;
    constexpr auto rhs = 4_c;
    constexpr bounded<16> expected = 3 * 4;
    constexpr auto result = lhs * rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // constant * bounded -> bounded
  {
    constexpr auto lhs = 3_c;
    constexpr bounded<4> rhs = 4;
    constexpr bounded<12> expected = 3 * 4;
    constexpr auto result = lhs * rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded / constant -> bounded
  {
    constexpr bounded<10> lhs = 10;
    constexpr auto rhs = 3_c;
    constexpr bounded<10> expected = 10 / 3;
    constexpr auto result = lhs / rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // constant / bounded -> int
  {
    constexpr auto lhs = 10_c;
    constexpr bounded<10> rhs = 3;
    constexpr int expected = 10 / 3;
    constexpr auto result = lhs / rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // bounded % constant -> bounded
  {
    constexpr bounded<10> lhs = 9;
    constexpr auto rhs = 3_c;
    constexpr bounded<2> expected = 9 % 3;
    constexpr auto result = lhs % rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }

  // constant % bounded -> bounded
  {
    constexpr auto lhs = 9_c;
    constexpr bounded<10> rhs = 3;
    constexpr bounded<9> expected = 9 % 3;
    constexpr auto result = lhs % rhs;

    static_assert(expected == result);
    static_assert(std::same_as<decltype(expected), decltype(result)>);
  }
}

