#pragma once

#include "../detail/prologue.hpp"

#include <future>
#include <utility>
#include <type_traits>


namespace ubu
{

namespace detail
{


template<class C, class... Cs>
concept has_because_of_member_function = requires(C c, Cs... cs)
{
  c.because_of(cs...);
};

template<class C, class... Cs>
concept has_because_of_free_function = requires(C c, Cs... cs)
{
  because_of(c,cs...);
};

template<class C, class... Cs>
concept has_because_of_customization = 
  has_because_of_member_function<C,Cs...> or
  has_because_of_free_function<C,Cs...>
;


struct dispatch_because_of
{
  template<class C, class... Cs>
    requires has_because_of_member_function<C&&,Cs&&...>
  constexpr auto operator()(C&& c, Cs&&... cs) const
  {
    return std::forward<C>(c).because_of(std::forward<Cs>(cs)...);
  }

  template<class C, class... Cs>
    requires (!has_because_of_member_function<C&&,Cs&&...> and
               has_because_of_free_function<C&&,Cs&&...>)
  constexpr auto operator()(C&& c, Cs&&... cs) const
  {
    return because_of(std::forward<C>(c), std::forward<Cs>(cs)...);
  }


  // a single cause 
  template<class C>
    requires (!has_because_of_member_function<C&&> and
              !has_because_of_free_function<C&&> and
              std::constructible_from<std::remove_cvref_t<C>, C&&>)
  constexpr std::remove_cvref_t<C> operator()(C&& cause) const
  {
    return std::forward<C>(cause);
  }


  // at least three causes
  template<class C1, class C2, class C3, class... Cs>
    requires (!has_because_of_member_function<C1&&,C2&&,C3&&,Cs&&...> and
              !has_because_of_free_function<C1&&,C2&&,C3&&,Cs&&...> and
              has_because_of_customization<C1&&,C2&&>)
  constexpr auto operator()(C1&& c1, C2&& c2, C3&& c3, Cs&&... cs) const
  {
    // combine c1 and c2
    auto c1_and_c2 = (*this)(std::forward<C1>(c1), std::forward<C2>(c2));

    // recurse with the combined result
    return (*this)(std::move(c1_and_c2), std::forward<Cs>(cs)...);
  }


  // customization for std::future<void>
  // XXX rather than provide this customization for std::future<void>, maybe we should just implement
  // a bare-bones c++ event type using standard synchronization primitives and allow std::future<void>
  // to be created from that
  inline std::future<void> operator()(std::future<void>&& f1, std::future<void>&& f2) const
  {
    return std::async(std::launch::deferred, [f1 = std::move(f1), f2 = std::move(f2)]
    {
      f1.wait();
      f2.wait();
    });
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_because_of because_of;

} // end anonymous namespace


template<class C, class... Cs>
using because_of_result_t = decltype(ubu::because_of(std::declval<C>(), std::declval<Cs>()...));


} // end ubu


#include "../detail/epilogue.hpp"

