#pragma once

#include "../detail/prologue.hpp"

#include "../event/wait.hpp"
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class E, class C>
concept has_contingent_on_all_member_function = requires(E executor, C events)
{
  executor.contingent_on_all(events);
};

template<class E, class C>
concept has_contingent_on_all_free_function = requires(E executor, C events)
{
  contingent_on_all(executor, events);
};


template<class ContainerOfEvent>
class bulk_event
{
  public:
    bulk_event(const bulk_event&) = delete;

    bulk_event(bulk_event&&) = default;

    bulk_event(ContainerOfEvent&& events)
      : events_{std::move(events)}
    {}

    void wait()
    {
      for(auto& e : events_)
      {
        ASPERA_NAMESPACE::wait(e);
      }
    }

  private:
    ContainerOfEvent events_;
};


// this is the type of contingent_on_all
struct dispatch_contingent_on_all
{
  // this dispatch path calls the member function
  template<class E, class C>
    requires has_contingent_on_all_member_function<E&&,C&&>
  constexpr auto operator()(E&& executor, C&& events) const
  {
    return std::forward<E>(executor).contingent_on_all(std::forward<C>(events));
  }

  // this dispatch path calls the free function
  template<class E, class C>
    requires (!has_contingent_on_all_member_function<E&&,C&&> and has_contingent_on_all_free_function<E&&,C&&>)
  constexpr auto operator()(E&& executor, C&& events) const
  {
    return contingent_on_all(std::forward<E>(executor), std::forward<C>(events));
  }

  template<class E, class C> 
    requires (!has_contingent_on_all_member_function<E&&,C&&> and !has_contingent_on_all_free_function<E&&,C&&>)
  constexpr auto operator()(E&&, C&& events) const
  {
    return bulk_event{std::forward<C>(events)};
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_contingent_on_all contingent_on_all;

} // end anonymous namespace


template<class E, class C>
using contingent_on_all_result_t = decltype(ASPERA_NAMESPACE::contingent_on_all(std::declval<E>, std::declval<C>));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

