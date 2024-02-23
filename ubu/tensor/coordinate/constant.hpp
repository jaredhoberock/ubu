#pragma once

#include "../../detail/prologue.hpp"
#include "concepts/coordinate.hpp"
#include "detail/tuple_algorithm.hpp"
#include "zeros.hpp"
#include <concepts>
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
              and v < std::tuple_size_v<std::remove_cvref_t<T>>
              and requires(T&& tuple) { get<v>(std::forward<T>(tuple)); })
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

// parse_int_digits takes a variadic number of digits and converts them into an int
template<std::integral R, std::same_as<int>... Ts>
constexpr R parse_int_digits(R result, int digit, Ts... digits) noexcept
{
  if constexpr (sizeof...(Ts) == 0)
  {
    return 10 * result + digit;
  }
  else
  {
    return parse_int_digits<R>(10 * result + digit, digits...);
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
constexpr constant<detail::parse_int_digits<int>(0, (digits - '0')...)> operator "" _c() noexcept
{
  static_assert((('0' <= digits) && ...) && ((digits <= '9') && ...),
              "Expected 0 <= digit <= 9 for each digit of the integer.");
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

