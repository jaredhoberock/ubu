#pragma once

#include "../../../detail/prologue.hpp"

#include <concepts>

namespace ubu
{

namespace detail
{


template<class A>
concept has_make_null_address_static_member_function = requires
{
  { A::make_null_address() } -> std::same_as<A>;
};


// this is the type of make_null_address
template<class A>
struct dispatch_make_null_address
{
  template<class = void>
    requires has_make_null_address_static_member_function<A>
  constexpr A operator()() const
  {
    return A::make_null_address();
  }
};

// specialization for pointer types
template<class T>
struct dispatch_make_null_address<T*>
{
  constexpr T* operator()() const
  {
    return {};
  }
};


} // end detail


namespace
{

template<class T>
constexpr detail::dispatch_make_null_address<T> make_null_address;

} // end anonymous namespace


} // end ubu

#include "../../../detail/epilogue.hpp"

