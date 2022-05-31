#pragma once

#include "../detail/prologue.hpp"

#include <future>
#include <utility>
#include <type_traits>

UBU_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class... Es>
concept has_make_dependent_event_member_function = requires(E e, Es... es)
{
  e.make_dependent_event(es...);
};

template<class E, class... Es>
concept has_make_dependent_event_free_function = requires(E e, Es... es)
{
  make_dependent_event(e,es...);
};

template<class E, class... Es>
concept has_make_dependent_event_customization = 
  has_make_dependent_event_member_function<E,Es...> or
  has_make_dependent_event_free_function<E,Es...>
;


struct dispatch_make_dependent_event
{
  template<class E, class... Es>
    requires has_make_dependent_event_member_function<E&&,Es&&...>
  constexpr auto operator()(E&& e, Es&&... es) const
  {
    return std::forward<E>(e).make_dependent_event(std::forward<Es>(es)...);
  }

  template<class E, class... Es>
    requires (!has_make_dependent_event_member_function<E&&,Es&&...> and
               has_make_dependent_event_free_function<E&&,Es&&...>)
  constexpr auto operator()(E&& e, Es&&... es) const
  {
    return make_dependent_event(std::forward<E>(e), std::forward<Es>(es)...);
  }


  // a single event 
  template<class E>
    requires (!has_make_dependent_event_member_function<E&&> and
              !has_make_dependent_event_free_function<E&&> and
              std::constructible_from<std::remove_cvref_t<E>, E&&>)
  constexpr std::remove_cvref_t<E> operator()(E&& e) const
  {
    return std::forward<E>(e);
  }


  // at least three events
  template<class E1, class E2, class E3, class... Es>
    requires (!has_make_dependent_event_member_function<E1&&,E2&&,E3&&,Es&&...> and
              !has_make_dependent_event_free_function<E1&&,E2&&,E3&&,Es&&...> and
              has_make_dependent_event_customization<E1&&,E2&&>)
  constexpr auto operator()(E1&& e1, E2&& e2, E3&& e3, Es&&... es) const
  {
    // combine e1 and e2
    auto e1_and_e2 = (*this)(std::forward<E1>(e1), std::forward<E2>(e2));

    // recurse with the combined result
    return (*this)(std::move(e1_and_e2), std::forward<Es>(es)...);
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

constexpr detail::dispatch_make_dependent_event make_dependent_event;

} // end anonymous namespace


template<class E, class... Es>
using make_dependent_event_result_t = decltype(UBU_NAMESPACE::make_dependent_event(std::declval<E>(), std::declval<Es>()...));


UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

