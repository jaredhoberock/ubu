#pragma once

#include "../detail/prologue.hpp"

#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class P>
concept has_destroy_at_member_function = requires(P ptr)
{
  ptr.destroy_at();
};

template<class P>
concept has_destroy_at_free_function = requires(P ptr)
{
  destroy_at(ptr);
};


struct dispatch_destroy_at
{
  template<class P>
    requires has_destroy_at_member_function<P&&>
  constexpr decltype(auto) operator()(P&& p) const
  {
    return std::forward<P>(p).destroy_at();
  }

  template<class P>
    requires (!has_destroy_at_member_function<P&&> and
               has_destroy_at_free_function<P&&>)
  constexpr decltype(auto) operator()(P&& p) const
  {
    return destroy_at(std::forward<P>(p));
  }

  template<class T>
    requires std::is_trivially_destructible_v<T>
  constexpr void operator()(T*) const
  {
    // no-op
  }

  template<class T>
    requires (not std::is_trivially_destructible_v<T> and std::destructible<T>)
  constexpr void operator()(T* p) const
  {
    p->~T();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy_at destroy_at;

} // end anonymous namespace


template<class P>
using destroy_at_result_t = decltype(ASPERA_NAMESPACE::destroy_at(std::declval<P>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"


