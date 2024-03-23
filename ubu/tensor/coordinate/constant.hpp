#pragma once

#include "../../detail/prologue.hpp"
#include "concepts/coordinate.hpp"
#include "detail/tuple_algorithm.hpp"
#include "zeros.hpp"
#include <concepts>
#include <cstdint>
#include <limits>
#include <iostream>

namespace ubu
{

template<auto v>
struct constant
{
  using value_type = decltype(v);
  constexpr static value_type value = v;
  
  // conversion to value_type
  constexpr operator value_type () const noexcept
  {
    return value;
  }

  // element(tuple, constant) indexes a tuple-like
  template<detail::tuple_like T>
    requires (std::integral<decltype(v)>
              and v < std::tuple_size_v<std::remove_cvref_t<T>>)
  friend constexpr decltype(auto) element(T&& tuple, constant)
  {
    return get<v>(std::forward<T>(tuple));
  }

  // element(array, constant) indexes an array-like (that is not also a tuple-like)
  template<class A>
    requires (not detail::tuple_like<A>
              and requires(A&& array) { std::forward<A>(array)[v]; })
  friend constexpr decltype(auto) element(A&& array, constant)
  {
    return std::forward<A>(array)[v];
  }

  // element(number, constant) returns the number when the constant value is 0
  template<class N>
    requires (v == 0 and std::integral<std::remove_cvref_t<N>>)
  friend constexpr decltype(auto) element(N&& number, constant)
  {
    return std::forward<N>(number);
  }

  friend std::ostream& operator<<(std::ostream& os, constant)
  {
    return os << value;
  }

  // unary operators
  constexpr constant<+v> operator+() const noexcept { return {}; }
  constexpr constant<-v> operator-() const noexcept { return {}; }
  constexpr constant<!v> operator!() const noexcept { return {}; }

  // binary operators against another constant
#define CONSTANT_BIN_OP_CONSTANT(OP)\
  template<auto other> requires requires { v OP other; }\
  constexpr constant<(v OP other)> operator OP (constant<other>) const noexcept { return {}; }

  // relational operators
  CONSTANT_BIN_OP_CONSTANT(==)
  CONSTANT_BIN_OP_CONSTANT(!=)
  CONSTANT_BIN_OP_CONSTANT(<)
  CONSTANT_BIN_OP_CONSTANT(<=)
  CONSTANT_BIN_OP_CONSTANT(>)
  CONSTANT_BIN_OP_CONSTANT(>=)

  // arithmetic operators
  CONSTANT_BIN_OP_CONSTANT(+)
  CONSTANT_BIN_OP_CONSTANT(-)
  CONSTANT_BIN_OP_CONSTANT(*)
  CONSTANT_BIN_OP_CONSTANT(|)
  CONSTANT_BIN_OP_CONSTANT(^)
  CONSTANT_BIN_OP_CONSTANT(<<)
  CONSTANT_BIN_OP_CONSTANT(>>)

  // operator/ has an additional requirement that the denominator is not zero
  template<auto other> requires (other != 0) and requires { v / other; }
  constexpr constant<(v / other)> operator/(constant<other>) const noexcept { return {}; }

  // operator% has an additional requirement that the denominator is not zero
  template<auto other> requires (other != 0) and requires { v % other; }
  constexpr constant<(v % other)> operator%(constant<other>) const noexcept { return {}; }

#undef CONSTANT_BIN_OP_CONSTANT

  // operators for dynamic values are handled via conversion to value_type
};


// specialize zeros<constant<v>>
template<auto v>
  requires coordinate<decltype(v)>
constexpr auto zeros<constant<v>> = zeros<decltype(v)>;

template<auto v>
  requires coordinate<decltype(v)>
constexpr auto zeros<constant<v>&> = zeros<decltype(v)>;

template<auto v>
  requires coordinate<decltype(v)>
constexpr auto zeros<const constant<v>&> = zeros<decltype(v)>;


#if defined(__cpp_user_defined_literals)

namespace detail
{

template<std::same_as<std::uint64_t>... Ts>
constexpr std::uint64_t as_uint64_t(std::uint64_t result, std::uint64_t base, std::uint64_t first_digit, Ts... rest_digits) noexcept
{
  if constexpr (sizeof...(rest_digits) == 0)
  {
    return result * base + first_digit;
  }
  else
  {
    return as_uint64_t(result * base + first_digit, base, rest_digits...);
  }
}

template<char c>
constexpr bool is_decimal_digit() noexcept
{
  return c >= '0' and c <= '9';
}

template<char c>
constexpr std::uint64_t to_decimal_digit() noexcept
{
  return c - '0';
}

template<char c>
constexpr bool is_octal_digit() noexcept
{
  return c >= '0' and c <= '7';
}

template<char c>
constexpr std::uint64_t to_octal_digit() noexcept
{
  return c - '0';
}

template<char c>
constexpr bool is_hexadecimal_digit() noexcept
{
  return (c >= '0' and c <= '9') or (c >= 'A' and c <= 'F') or (c >= 'a' and c <= 'f');
}

template<char c>
constexpr std::uint64_t to_hexadecimal_digit() noexcept
{
  return (c >= '0' and c <= '9') ? c - '0' :
         (c >= 'A' and c <= 'F') ? 10 + c - 'A' :
         10 + c - 'a';
}


// this helper skips the "0x" prefix of a hexadecimal number
template<char prefix_0, char prefix_1, char... chars>
constexpr std::uint64_t parse_hexadecimal_integer() noexcept
{
  static_assert(sizeof...(chars) > 0, "Hexadecimal number must have at least one digit.");
  static_assert((... and is_hexadecimal_digit<chars>()), "Invalid hexadecimal number.");
  return as_uint64_t(0, 16, to_hexadecimal_digit<chars>()...);
}

// this helper skips the "0" prefix of an octal number
template<char prefix, char... chars>
constexpr std::uint64_t parse_octal_integer() noexcept
{
  static_assert(sizeof...(chars) > 0, "Octal number must have at least one digit.");
  static_assert((... and is_octal_digit<chars>()), "Invalid octal number.");
  return as_uint64_t(0, 8, to_octal_digit<chars>()...);
}

template<char... chars>
constexpr std::uint64_t parse_decimal_integer() noexcept
{
  static_assert(sizeof...(chars) > 0, "Decimal number must have at least one digit.");
  static_assert((... and is_decimal_digit<chars>()), "Invalid decimal number.");
  return as_uint64_t(0, 10, to_decimal_digit<chars>()...);
}


template<std::uint64_t v>
constexpr std::integral auto narrow_decimal_integer() noexcept
{
  if constexpr (v <= std::numeric_limits<int>::max())
  {
    return static_cast<int>(v);
  }
  else if constexpr (v <= std::numeric_limits<long int>::max())
  {
    return static_cast<long int>(v);
  }
  else
  {
    return static_cast<long long int>(v);
  }
}

template<std::uint64_t v>
constexpr std::integral auto narrow_nondecimal_integer() noexcept
{
  if constexpr (v <= std::numeric_limits<int>::max())
  {
    return static_cast<int>(v);
  }
  else if constexpr (v <= std::numeric_limits<unsigned int>::max())
  {
    return static_cast<unsigned int>(v);
  }
  else if constexpr (v <= std::numeric_limits<long int>::max())
  {
    return static_cast<long int>(v);
  }
  else if constexpr (v <= std::numeric_limits<unsigned long int>::max())
  {
    return static_cast<unsigned long int>(v);
  }
  else if constexpr (v <= std::numeric_limits<long long int>::max())
  {
    return static_cast<long long int>(v);
  }
  else
  {
    return static_cast<unsigned long long int>(v);
  }
}

template<char... chars>
constexpr auto parse_integer_literal() noexcept 
{
  constexpr char string[] = {chars...};

  if constexpr (sizeof...(chars) > 2 and string[0] == '0' and (string[1] == 'x' or string[1] == 'X'))
  {
    constexpr auto result = parse_hexadecimal_integer<chars...>();
    return narrow_nondecimal_integer<result>();
  }
  else if constexpr (sizeof...(chars) > 1 and string[0] == '0')
  {
    constexpr auto result = parse_octal_integer<chars...>();
    return narrow_nondecimal_integer<result>();
  }
  else
  {
    constexpr auto result = parse_decimal_integer<chars...>();
    return narrow_decimal_integer<result>();
  }
}

} // end detail


// user-defined literal operator allows constant written as literals. For example,
//
//   using namespace ubu;
//
//   auto var = 32_c;
//
// var has type constant<32>.
template<char... digits>
constexpr constant<detail::parse_integer_literal<digits...>()> operator "" _c() noexcept
{
  return {};
}

#endif // __cpp_user_defined_literals

} // end ubu

#if __has_include(<fmt/format.h>)

#include <fmt/format.h>
#include <fmt/color.h>

template<auto v>
struct fmt::formatter<ubu::constant<v>>
{
  template<class ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template<class FormatContext>
  auto format(const ubu::constant<v>& c, FormatContext& ctx)
  {
    return fmt::format_to(ctx.out(), fmt::emphasis::bold, "{}_c", c.value);
  }
};

#endif // __has_include

#include "../../detail/epilogue.hpp"

