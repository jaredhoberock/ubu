#pragma once

#include "../detail/prologue.hpp"

#include "../coordinate/lattice.hpp"
#include "../event/event.hpp"
#include "contingent_on_all.hpp"
#include "executor.hpp"
#include "executor_event.hpp"
#include "then_execute.hpp"

#include <utility>
#include <vector>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class Ev, class S, class F>
concept has_bulk_execute_member_function = requires(Ex executor, Ev event, S grid_shape, F function) { executor.bulk_execute(event, grid_shape, function); };

template<class Ex, class Ev, class S, class F>
concept has_bulk_execute_free_function = requires(Ex executor, Ev event, S grid_shape, F function) { bulk_execute(executor, event, grid_shape, function); };


// this is the type of bulk_execute
struct dispatch_bulk_execute
{
  // this dispatch path calls the member function
  template<class Ex, class Ev, class S, class F>
    requires has_bulk_execute_member_function<Ex&&,Ev&&,S&&,F&&>
  constexpr auto operator()(Ex&& executor, Ev&& event, S&& grid_shape, F&& function) const
  {
    return std::forward<Ex>(executor).bulk_execute(std::forward<Ev>(event), std::forward<S>(grid_shape), std::forward<F>(function));
  }

  // this dispatch path calls the free function
  template<class Ex, class Ev, class S, class F>
    requires (!has_bulk_execute_member_function<Ex&&,Ev&&,S&&,F&&> and has_bulk_execute_free_function<Ex&&,Ev&&,S&&,F&&>)
  constexpr auto operator()(Ex&& executor, Ev&& event, S&& grid_shape, F&& function) const
  {
    return bulk_execute(std::forward<Ex>(executor), std::forward<Ev>(event), std::forward<S>(grid_shape), std::forward<F>(function));
  }

  // XXX another default path with higher priority than below should call bulk_execute
  //     with the executor's coordinate_type and map indicies from the native executor coordinate
  //     to our requested space


  template<executor Ex, event Ev, grid_coordinate S, std::regular_invocable<S> F>
    requires (!has_bulk_execute_member_function<Ex&&,Ev&&,S&&,F&&> and !has_bulk_execute_free_function<Ex&&,Ev&&,S&&,F&&>)
  auto operator()(const Ex& ex, Ev&& before, S&& grid_shape, F function) const
  {
    // XXX this should maybe be vector<then_execute_result_t<lambda>>
    //     should maybe get an allocator out of the executor or something
    std::vector<executor_event_t<Ex>> events;

    for(auto coord : lattice{std::forward<S>(grid_shape)})
    {
      events.push_back(then_execute(ex, before, [function,coord]
      {
        std::invoke(function, coord);
      }));
    }

    return contingent_on_all(ex, std::move(events));
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_bulk_execute bulk_execute;

} // end anonymous namespace


template<class Ex, class Ev, class S, class F>
using bulk_execute_result_t = decltype(ASPERA_NAMESPACE::bulk_execute(std::declval<Ex>(), std::declval<Ev>(), std::declval<S>(), std::declval<F>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

