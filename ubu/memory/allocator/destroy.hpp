#pragma once

#include "../../detail/prologue.hpp"

#include "../destroy_at.hpp"

#include <type_traits>
#include <utility>

UBU_NAMESPACE_OPEN_BRACE

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

  // default path for pointers calls destroy_at
  template<class A, class P,
           class T = typename std::pointer_traits<std::remove_cvref_t<P>>::element_type
          >
    requires (!has_destroy_member_function<A&&, P&&> and
              !has_destroy_free_function<A&&, P&&> and
              !std::is_void_v<T>)
  constexpr void operator()(A&&, P&& ptr) const
  {
    destroy_at(std::forward<P>(ptr));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_destroy destroy;

} // end anonymous namespace


template<class A, class P>
using destroy_result_t = decltype(UBU_NAMESPACE::destroy(std::declval<A>(), std::declval<P>()));


UBU_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

