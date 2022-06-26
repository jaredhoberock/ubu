#pragma once

#include "../../detail/prologue.hpp"

#include "../../event/happening.hpp"
#include "../allocator/asynchronous_allocator.hpp"
#include "../allocator/deallocate_after.hpp"
#include "../allocator/destroy_after.hpp"
#include <utility>


namespace ubu
{

namespace detail
{


// XXX this should have a path that attempts a rebind


template<class A, class H, class P, class N>
concept has_delete_after_member_function = requires(A alloc, H before, P ptr, N n)
{
  {alloc.delete_after(before, ptr, n)} -> happening;
};


template<class A, class H, class P, class N>
concept has_delete_after_free_function = requires(A alloc, H before, P ptr, N n)
{
  {delete_after(alloc, before, ptr, n)} -> happening;
};


// this is the type of delete_after
struct dispatch_delete_after
{
  // this dispatch path calls the member function
  template<class Allocator, class Happening, class P, class N>
    requires has_delete_after_member_function<Allocator&&, Happening&&, P&&, N&&>
  constexpr auto operator()(Allocator&& alloc, Happening&& before, P&& ptr, N&& n) const
  {
    return std::forward<Allocator>(alloc).delete_after(std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // this dispatch path calls the free function
  template<class Allocator, class Happening, class P, class N>
    requires (!has_delete_after_member_function<Allocator&&, Happening&&, P&&, N&&> and
               has_delete_after_free_function<Allocator&&, Happening&&, P&&, N&&>)
  constexpr auto operator()(Allocator&& alloc, Happening&& before, P&& ptr, N&& n) const
  {
    return delete_after(std::forward<Allocator>(alloc), std::forward<Happening>(before), std::forward<P>(ptr), std::forward<N>(n));
  }

  // the default path
  //   1. calls destroy_after
  //   2. calls deallocate_after
  //
  //   XXX N should be allocator_traits<A>::size_type
  template<pointer_like P, asynchronous_allocator_of<pointer_pointee_t<P>> A, happening H, class N>
    requires (!has_delete_after_member_function<A&&, H&&, P, N> and
              !has_delete_after_free_function<A&&, H&&, P, N> and
              executor_associate<A&&>)
  constexpr auto operator()(A&& alloc, H&& before, P ptr, N n) const
  {
    // destroy
    auto after_destructors = ubu::destroy_after(std::forward<A>(alloc), std::forward<H>(before), ptr, n);

    // deallocate
    return deallocate_after(std::forward<A>(alloc), std::move(after_destructors), ptr, n);
  }

};


} // end detail


namespace
{

constexpr detail::dispatch_delete_after delete_after;

} // end anonymous namespace


template<class A, class H, class P, class N>
using delete_after_result_t = decltype(ubu::delete_after(std::declval<A>(), std::declval<H>(), std::declval<P>(), std::declval<N>()));


} // end ubu

#include "../../detail/epilogue.hpp"

