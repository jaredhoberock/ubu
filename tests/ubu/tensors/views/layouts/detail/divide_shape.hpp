#include <cassert>
#include <fmt/format.h>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/layouts/detail/divide_shape.hpp>

void test_divide_shape()
{
  {
    int numerator = 7;
    int denominator = 13;

    auto [quotient, divisor] = ubu::detail::divide_shape(numerator, denominator);

    int expected_quotient = 7;
    int expected_divisor = 13;

    assert(expected_quotient == quotient);
    assert(expected_divisor == divisor);
  }

  {
    int numerator = 13;
    int denominator = 7;

    auto [quotient, divisor] = ubu::detail::divide_shape(numerator, denominator);
    
    int expected_quotient = 13;
    int expected_divisor = 7;

    assert(expected_quotient == quotient);
    assert(expected_divisor == divisor);
  }

  {
    std::pair numerator(3,4);
    int denominator = 2;

    auto [quotient, divisor] = ubu::detail::divide_shape(numerator, denominator);
    
    std::pair expected_quotient(2,4);
    std::pair expected_divisor(2,1);

    assert(expected_quotient == quotient);
    assert(expected_divisor == divisor);
  }

  {
    std::tuple numerator(std::make_pair(2,3), std::make_tuple(4,5,6));
    int denominator = 3;

    auto [quotient, divisor] = ubu::detail::divide_shape(numerator, denominator);
    
    std::tuple expected_quotient(std::make_pair(1,2), std::make_tuple(4,5,6));
    std::tuple expected_divisor(std::make_pair(2,2), std::make_tuple(1,1,1));

    assert(expected_quotient == quotient);
    assert(expected_divisor == divisor);
  }

  {
    std::tuple numerator(4, std::make_pair(2,3), std::make_tuple(4, std::make_pair(5,6), 7));
    int denominator = 12;

    auto [quotient, divisor] = ubu::detail::divide_shape(numerator, denominator);

    std::tuple expected_quotient(1, std::make_pair(1,2), std::make_tuple(4, std::make_pair(5,1), 7));
    std::tuple expected_divisor(4, std::make_pair(2,2), std::make_tuple(1, std::make_pair(6,1), 1));

    assert(expected_quotient == quotient);
    assert(expected_divisor == divisor);
  }
}
