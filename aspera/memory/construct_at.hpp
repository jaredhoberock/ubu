#pragma once

#include "../detail/prologue.hpp"

#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class P, class... Args>
concept has_construct_at_member_function = requires(P ptr, Args... args)
{
  ptr.construct_at(args...);
};

template<class P, class... Args>
concept has_construct_at_free_function = requires(P ptr, Args... args)
{
  construct_at(ptr, args...);
};


struct dispatch_construct_at
{
  template<class P, class... Args>
    requires has_construct_at_member_function<P&&,Args&&...>
  constexpr decltype(auto) operator()(P&& p, Args&&... args) const
  {
    return std::forward<P>(p).construct_at(std::forward<Args>(args)...);
  }

  template<class P, class... Args>
    requires (!has_construct_at_member_function<P&&,Args&&...> and
               has_construct_at_free_function<P&&,Args&&...>)
  constexpr decltype(auto) operator()(P&& p, Args&&... args) const
  {
    return construct_at(std::forward<P>(p), std::forward<Args>(args)...);
  }

  template<class T, class... Args>
    requires std::constructible_from<T,Args&&...>
  constexpr void operator()(T* p, Args&&... args) const
  {
    new(p) T(std::forward<Args>(args)...);
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_construct_at construct_at;

} // end anonymous namespace


template<class P, class... Args>
using construct_at_result_t = decltype(ASPERA_NAMESPACE::construct_at(std::declval<P>(), std::declval<Args>()...));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

