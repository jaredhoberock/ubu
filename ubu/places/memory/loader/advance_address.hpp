#pragma once

#include "../../../detail/prologue.hpp"

#include <cstdint>
#include <utility>

namespace ubu
{

namespace detail
{


template<class A, class N>
concept has_advance_address_member_function = requires(A a, N n)
{
  a.advance_address(n);
};


template<class A, class N>
concept has_advance_address_free_function = requires(A a, N n)
{
  advance_address(a, n);
};


// this is the type of advance_address
struct dispatch_advance_address
{
  template<class Address, class N>
    requires has_advance_address_member_function<Address&&,N&&>
  constexpr auto operator()(Address&& a, N&& n) const
  {
    return std::forward<Address>(a).advance_address(std::forward<N>(n));
  }

  template<class Address, class N>
    requires (!has_advance_address_member_function<Address&&,N&&> and
              has_advance_address_free_function<Address&&,N&&>)
  constexpr auto operator()(Address&& a, N&& n) const
  {
    return advance_address(std::forward<Address>(a), std::forward<N>(n));
  }

  // default paths for void pointers
  void operator()(void*& ptr, std::ptrdiff_t n) const
  {
    char* ptr_to_char = reinterpret_cast<char*>(ptr);
    ptr_to_char += n;
    ptr = reinterpret_cast<void*>(ptr_to_char);
  }

  void operator()(const void*& ptr, std::ptrdiff_t n) const
  {
    const char* ptr_to_char = reinterpret_cast<const char*>(ptr);
    ptr_to_char += n;
    ptr = reinterpret_cast<const void*>(ptr_to_char);
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_advance_address advance_address;

} // end anonymous namespace


} // end ubu

#include "../../../detail/epilogue.hpp"

