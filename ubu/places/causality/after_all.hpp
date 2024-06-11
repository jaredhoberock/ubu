#pragma once

#include "../../detail/prologue.hpp"

#include "happening.hpp"
#include <future>
#include <utility>
#include <type_traits>


namespace ubu
{

namespace detail
{


template<class B, class... Bs>
concept has_after_all_member_function = requires(B before, Bs... befores)
{
  { before.after_all(befores...) } -> happening;
};

template<class B, class... Bs>
concept has_after_all_free_function = requires(B before, Bs... befores)
{
  { after_all(before,befores...) } -> happening;
};

template<class B, class... Bs>
concept has_after_all_customization = 
  has_after_all_member_function<B,Bs...> or
  has_after_all_free_function<B,Bs...>
;


struct dispatch_after_all
{
  template<class B, class... Bs>
    requires has_after_all_member_function<B&&,Bs&&...>
  constexpr happening auto operator()(B&& before, Bs&&... befores) const
  {
    return std::forward<B>(before).after_all(std::forward<Bs>(befores)...);
  }

  template<class B, class... Bs>
    requires (    not has_after_all_member_function<B&&,Bs&&...>
              and has_after_all_free_function<B&&,Bs&&...>)
  constexpr happening auto operator()(B&& before, Bs&&... befores) const
  {
    return after_all(std::forward<B>(before), std::forward<Bs>(befores)...);
  }


  // a single happening 
  template<happening B>
    requires (    not has_after_all_member_function<B&&>
              and not has_after_all_free_function<B&&>
              and std::constructible_from<std::remove_cvref_t<B>, B&&>)
  constexpr std::remove_cvref_t<B> operator()(B&& before) const
  {
    return std::forward<B>(before);
  }


  // at least three causes
  template<happening B1, happening B2, happening B3, happening... Bs>
    requires (    not has_after_all_member_function<B1&&,B2&&,B3&&,Bs&&...>
              and not has_after_all_free_function<B1&&,B2&&,B3&&,Bs&&...>
              and has_after_all_customization<B1&&,B2&&>)
  constexpr happening auto operator()(B1&& before1, B2&& before2, B3&& before3, Bs&&... befores) const
  {
    // combine before1 and before2
    auto before1_and_before2 = (*this)(std::forward<B1>(before1), std::forward<B2>(before2));

    // recurse with the combined result
    return (*this)(std::move(before1_and_before2), std::forward<Bs>(befores)...);
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

constexpr detail::dispatch_after_all after_all;

} // end anonymous namespace


template<class C, class... Cs>
using after_all_result_t = decltype(after_all(std::declval<C>(), std::declval<Cs>()...));


} // end ubu


#include "../../detail/epilogue.hpp"

