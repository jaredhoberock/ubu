#pragma once

#include "../../detail/prologue.hpp"

#include <type_traits>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class A, class P>
concept has_destroy_member_function = requires(A alloc, P ptr)
{
  alloc.destroy(ptr);
};


template<class A, class P>
concept has_destroy_free_function = requires(A alloc, P ptr)
{
  destroy(alloc, ptr);
};


// this is the type of destroy
struct dispatch_destroy
{
  // this dispatch path calls the member function
  template<class Alloc, class P>
    requires has_destroy_member_function<Alloc&&, P&&>
  constexpr auto operator()(Alloc&& alloc, P&& ptr) const
  {
    return std::forward<Alloc>(alloc).destroy(std::forward<P>(ptr));
  }

  // this dispatch path calls the free function
  template<class Alloc, class P>
    requires (!has_destroy_member_function<Alloc&&, P&&> and
               has_destroy_free_function<Alloc&&, P&&>)
  constexpr auto operator()(Alloc&& alloc, P&& ptr) const
  {
    return destroy(std::forward<Alloc>(alloc), std::forward<P>(ptr));
  }

  // default path for pointers simply calls the destructor inline
  template<class A, class T>
    requires (!has_destroy_member_function<A&&, T*> and
              !has_destroy_free_function<A&&, T*> and
              !std::is_void_v<T>)
  constexpr void operator()(A&&, T* ptr) const
  {
    // XXX consider calling destroy_at once it exists
    ptr->~T();
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy destroy;

} // end anonymous namespace


template<class A, class P>
using destroy_result_t = decltype(ASPERA_NAMESPACE::destroy(std::declval<A>(), std::declval<P>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

