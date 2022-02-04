#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/make_complete_event.hpp"
#include "allocate_after.hpp"
#include <memory>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class A, class N>
concept has_first_allocate_member_function = requires(A alloc, N n)
{
  // XXX this should check that the result is a future<pointer>
  alloc.first_allocate(n);
};


template<class A, class N>
concept has_first_allocate_free_function = requires(A alloc, N n)
{
  // XXX this should check that the result is a future<pointer>
  first_allocate(alloc, n);
};


// this is the type of first_allocate
struct dispatch_first_allocate
{
  // this dispatch path calls the member function
  template<class Alloc, class N>
    requires has_first_allocate_member_function<Alloc&&, N&&>
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return std::forward<Alloc>(alloc).first_allocate(std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class N>
    requires (!has_first_allocate_member_function<Alloc&&, N&&> and
               has_first_allocate_free_function<Alloc&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const
  {
    return first_allocate(std::forward<Alloc>(alloc), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Alloc, class N>
    requires (!has_first_allocate_member_function<Alloc&&, N&&> and
              !has_first_allocate_free_function<Alloc&&, N&&>)
  constexpr auto operator()(Alloc&& alloc, N&& n) const ->
    decltype(allocate_after(std::forward<Alloc>(alloc), make_complete_event(std::forward<Alloc>(alloc)), std::forward<N>(n)))
  {
    return allocate_after(std::forward<Alloc>(alloc), make_complete_event(std::forward<Alloc>(alloc)), std::forward<N>(n));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_first_allocate first_allocate;

} // end anonymous namespace


template<class A, class N>
using first_allocate_result_t = decltype(ASPERA_NAMESPACE::first_allocate(std::declval<A>(), std::declval<N>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

