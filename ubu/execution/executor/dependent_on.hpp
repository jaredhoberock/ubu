#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/because_of.hpp"
#include "../../causality/happening.hpp"
#include "../../causality/wait.hpp"
#include "executor.hpp"
#include <type_traits>
#include <tuple>
#include <utility>


namespace ubu
{

namespace detail
{


template<class E, class... Happenings>
concept has_dependent_on_member_function = requires(E executor, Happenings... happenings)
{
  { executor.dependent_on(happenings...) } -> happening;
};

template<class E, class... Happenings>
concept has_dependent_on_free_function = requires(E executor, Happenings... happenings)
{
  { dependent_on(executor, happenings...) } -> happening;
};


// this is the type of dependent_on
struct dispatch_dependent_on
{
  // this dispatch path calls the member function
  template<class E, class... Happenings>
    requires has_dependent_on_member_function<E&&,Happenings&&...>
  constexpr auto operator()(E&& executor, Happenings&&... happenings) const
  {
    return std::forward<E>(executor).dependent_on(std::forward<Happenings>(happenings)...);
  }

  // this dispatch path calls the free function
  template<class E, class... Happenings>
    requires (!has_dependent_on_member_function<E&&,Happenings&&...> and
              has_dependent_on_free_function<E&&,Happenings&&...>)
  constexpr auto operator()(E&& executor, Happenings&&... happenings) const
  {
    return dependent_on(std::forward<E>(executor), std::forward<Happenings>(happenings)...);
  }

  // the default path drops the executor and calls because_of
  template<executor E, happening... Happenings>
    requires (!has_dependent_on_member_function<E&&,Happenings&&...> and
              !has_dependent_on_free_function<E&&,Happenings&&...>)
  constexpr auto operator()(E&&, Happenings&&... happenings) const
  {
    return because_of(std::forward<Happenings>(happenings)...);
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_dependent_on dependent_on;

} // end anonymous namespace


template<class E, class... Happenings>
using dependent_on_result_t = decltype(ubu::dependent_on(std::declval<E>, std::declval<Happenings>...));


} // end ubu


#include "../../detail/epilogue.hpp"

