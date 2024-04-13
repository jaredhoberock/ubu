#pragma once

#include "../detail/prologue.hpp"
#include "../tensor/coordinate/constant.hpp"
#include <concepts>
#include <iostream>
#include <limits>
#include <type_traits>

namespace ubu
{

// XXX do we really require a non-negative bound?
template<std::integral auto b>
  requires (b >= 0)
struct bounded
{
  using value_type = decltype(b);

  constexpr static value_type bound = b;

  value_type value{};

  constexpr bounded(value_type value) noexcept
    : value{value}
  {}

  bounded() = default;

  bounded(const bounded&) = default;

  // this ctor is provided for CTAD.
  // The second parameter, interpreted as the bound, is otherwise ignored
  constexpr bounded(value_type value, constant<b>)
    : bounded(value)
  {}

  template<std::integral auto c>
    requires (0 <= c and c <= b)
  constexpr bounded(constant<c>)
    : bounded(c)
  {}

  // conversion to value_type
  constexpr operator value_type () const noexcept
  {
    return value;
  }

  friend std::ostream& operator<<(std::ostream& os, bounded self)
  {
    return os << self.value;
  }

  // unary operators
  constexpr bounded operator+() const noexcept { return *this; }
  constexpr bounded operator-() const noexcept { return -value; }
  constexpr bounded& operator++() noexcept { ++value; return *this; }
  constexpr bounded& operator--() noexcept { --value; return *this; }
  constexpr bounded operator++(int) noexcept { return value++; }
  constexpr bounded operator--(int) noexcept { return value--; }

  // binary operators against a value_type
#define BIN_OP_VALUE(OP)\
  template<std::integral RHS> constexpr auto operator OP (RHS rhs) const noexcept { return value OP rhs; }

    // relational operators
//  // XXX do we need these, or can we rely on conversion to value_type?
//  BIN_OP_VALUE(==)
//  BIN_OP_VALUE(!=)
//  BIN_OP_VALUE(<)
//  BIN_OP_VALUE(<=)
//  BIN_OP_VALUE(>)
//  BIN_OP_VALUE(>=)

  // arithmetic operators
  BIN_OP_VALUE(+)
  BIN_OP_VALUE(-)
  BIN_OP_VALUE(*)
  BIN_OP_VALUE(%)
  BIN_OP_VALUE(|)
  BIN_OP_VALUE(^)
  BIN_OP_VALUE(<<)
#undef BIN_OP_VALUE

  // division allows us to keep our bound
  constexpr bounded operator/(value_type rhs) const noexcept
  {
    return value / rhs;
  }

  // shift left allows us to keep our bound
  constexpr bounded operator>>(value_type rhs) const noexcept
  {
    return value >> rhs;
  }

  // modulus with a bounded yields a bounded
  friend constexpr bounded<bound-1> operator%(value_type lhs, bounded rhs) noexcept
  {
    return {lhs % rhs.value};
  }

  // binary operators against a bounded
  template<std::integral auto other_bound>
    requires (other_bound >= 0)
  constexpr bounded<bound + other_bound> operator+(bounded<other_bound> rhs) const noexcept
  {
    return {value + rhs.value};
  }

  template<std::integral auto other_bound>
    requires (other_bound >= 0)
  constexpr auto operator-(bounded<other_bound> rhs) const noexcept
  {
    if constexpr(other_bound <= bound)
    {
      return bounded<bound>(value - rhs.value);
    }
    else
    {
      return value - rhs.value;
    }
  }

  template<std::integral auto other_bound>
    requires (other_bound >= 0)
  constexpr bounded<bound * other_bound> operator*(bounded<other_bound> rhs) const noexcept
  {
    return {value * rhs.value};
  }

  template<std::integral auto other_bound>
    requires (other_bound >= 0)
  constexpr bounded<other_bound-1> operator%(bounded<other_bound> rhs) const noexcept
  {
    return {value % rhs.value};
  }

  // binary operators against a constant
  template<std::integral auto c>
    requires (bound + c >= 0)
  constexpr bounded<bound + c> operator+(constant<c>) const noexcept
  {
    return {value + c};
  }

  template<std::integral auto c>
    requires (bound + c >= 0)
  friend constexpr bounded<c + bound> operator+(constant<c>, bounded rhs) noexcept
  {
    return {c + rhs.value};
  }

  template<std::integral auto c>
  constexpr auto operator-(constant<c>) const noexcept
  {
    if constexpr (0 <= c and c <= bound)
    {
      return bounded<bound - c>(value - c);
    }
    else
    {
      return value - c;
    }
  }

  template<std::integral auto c>
    requires (bound * c >= 0)
  constexpr bounded<bound * c> operator*(constant<c>) const noexcept
  {
    return {value * c};
  }

  template<std::integral auto c>
    requires (bound * c >= 0)
  friend constexpr bounded<c * bound> operator*(constant<c>, bounded rhs) noexcept
  {
    return {c * rhs.value};
  }

  template<std::integral auto c>
    requires (c > 0)
  constexpr bounded<c-1> operator%(constant<c>) const noexcept
  {
    return {value % c};
  }

  template<std::integral auto c>
  friend constexpr bounded<bound - 1> operator%(constant<c>, bounded rhs) noexcept
  {
    return {c % rhs.value};
  }
};

} // end ubu

template<std::integral auto b>
class std::numeric_limits<ubu::bounded<b>> : std::numeric_limits<decltype(b)>
{
  private:
    using limits = std::numeric_limits<decltype(b)>;

  public:
    static constexpr ubu::bounded<b> min() noexcept
    {
      return limits::min();
    }

    static constexpr ubu::bounded<b> lowest() noexcept
    {
      return limits::lowest();
    }

    static constexpr ubu::bounded<b> max() noexcept
    {
      return b;
    }

    static constexpr ubu::bounded<b> epsilon() noexcept
    {
      return limits::epsilon();
    }

    static constexpr ubu::bounded<b> round_error() noexcept
    {
      return limits::round_error();
    }

    static constexpr ubu::bounded<b> infinity() noexcept
    {
      return limits::infinity();
    }

    static constexpr ubu::bounded<b> quiet_NaN() noexcept
    {
      return limits::quiet_NaN();
    }

    static constexpr ubu::bounded<b> signaling_NaN() noexcept
    {
      return limits::signaling_NaN();
    }

    static constexpr ubu::bounded<b> denorm_min() noexcept
    {
      return limits::denorm_min();
    }
};

#include "../detail/epilogue.hpp"

