#pragma once

#include "../../detail/prologue.hpp"
#include "../construct_at.hpp"

#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class A, class P, class... Args>
concept has_construct_member_function = requires(A alloc, P ptr, Args... args)
{
  alloc.construct(ptr, args...);
};

template<class A, class P, class... Args>
concept has_construct_free_function = requires(A alloc, P ptr, Args... args)
{
  construct(alloc, ptr, args...);
};


struct dispatch_construct
{
  template<class Alloc, class P, class... Args>
    requires has_construct_member_function<Alloc&&,P&&,Args&&...>
  constexpr decltype(auto) operator()(Alloc&& a, P&& p, Args&&... args) const
  {
    return std::forward<Alloc>(a).construct(std::forward<P>(p), std::forward<Args>(args)...);
  }

  template<class Alloc, class P, class... Args>
    requires (!has_construct_member_function<Alloc&&,P&&,Args&&...> and
               has_construct_free_function<Alloc&&,P&&,Args&&...>)
  constexpr decltype(auto) operator()(Alloc&& a, P&& p, Args&&... args) const
  {
    return construct(std::forward<Alloc>(a), std::forward<P>(p), std::forward<Args>(args)...);
  }

  template<class Alloc, class P, class... Args>
    requires (!has_construct_member_function<Alloc&&,P&&,Args&&...> and
              !has_construct_free_function<Alloc&&,P&&,Args&&...>)
  constexpr void operator()(Alloc&&, P&& p, Args&&... args) const
  {
    construct_at(std::forward<P>(p), std::forward<Args>(args)...);
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_construct construct;

} // end anonymous namespace


template<class A, class P, class... Args>
using construct_result_t = decltype(ASPERA_NAMESPACE::construct(std::declval<A>(), std::declval<P>(), std::declval<Args>()...));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

