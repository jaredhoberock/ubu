#pragma once

#include "../../../detail/prologue.hpp"

#include <concepts>
#include <cstdint>
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class B>
concept has_address_difference_member_function = requires(A lhs, B rhs)
{
  { lhs.address_difference(rhs) } -> std::convertible_to<std::ptrdiff_t>;
};


template<class A, class B>
concept has_address_difference_free_function = requires(A lhs, B rhs)
{
  { address_difference(lhs, rhs) } -> std::convertible_to<std::ptrdiff_t>;
};


// this is the type of address_difference
struct dispatch_address_difference
{
  template<class A, class B>
    requires has_address_difference_member_function<A&&,B&&>
  constexpr auto operator()(A&& lhs, B&& rhs) const
  {
    return std::forward<A>(lhs).address_difference(std::forward<B>(rhs));
  }

  template<class A, class B>
    requires (!has_address_difference_member_function<A&&,B&&> and
              has_address_difference_free_function<A&&,B&&>)
  constexpr auto operator()(A&& lhs, B&& rhs) const
  {
    return address_difference(std::forward<A>(lhs), std::forward<B>(rhs));
  }

  // default path for void pointers
  decltype(auto) operator()(const void* lhs, const void* rhs) const
  {
    const char* lhs_char = reinterpret_cast<const char*>(lhs);
    const char* rhs_char = reinterpret_cast<const char*>(rhs);
    return lhs_char - rhs_char;
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_address_difference address_difference;

} // end anonymous namespace


template<class A, class B = A>
using address_difference_result_t = decltype(address_difference(std::declval<A>(), std::declval<B>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

