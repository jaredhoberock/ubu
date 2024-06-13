#pragma once

#include "../../../detail/prologue.hpp"

#include "../pointers/destroy_at.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

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
  template<class A, pointer_like P>
    requires (!has_destroy_member_function<A&&, P&&> and
              !has_destroy_free_function<A&&, P&&> and
              !std::is_void_v<pointer_pointee_t<P>>)
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
using destroy_result_t = decltype(ubu::destroy(std::declval<A>(), std::declval<P>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

