#pragma once

#include "../../detail/prologue.hpp"

#include "../../coordinate/colexicographic_index.hpp"
#include "../../coordinate/lattice.hpp"
#include "../../event/event.hpp"
#include "bulk_execution_grid.hpp"
#include "dependent_on.hpp"
#include "executor.hpp"
#include "executor_coordinate.hpp"
#include "executor_event.hpp"
#include "execute_after.hpp"

#include <utility>
#include <vector>


ASPERA_NAMESPACE_OPEN_BRACE

namespace detail
{


template<class Ex, class Ev, class S, class F>
concept has_bulk_execute_after_member_function = requires(Ex executor, Ev event, S grid_shape, F function) { executor.bulk_execute_after(event, grid_shape, function); };

template<class Ex, class Ev, class S, class F>
concept has_bulk_execute_after_free_function = requires(Ex executor, Ev event, S grid_shape, F function) { bulk_execute_after(executor, event, grid_shape, function); };


// this is the type of bulk_execute_after
struct dispatch_bulk_execute_after
{
  // this dispatch path calls the member function
  template<class Ex, class Ev, class S, class F>
    requires has_bulk_execute_after_member_function<Ex&&,Ev&&,S&&,F&&>
  constexpr auto operator()(Ex&& executor, Ev&& event, S&& grid_shape, F&& function) const
  {
    return std::forward<Ex>(executor).bulk_execute_after(std::forward<Ev>(event), std::forward<S>(grid_shape), std::forward<F>(function));
  }

  // this dispatch path calls the free function
  template<class Ex, class Ev, class S, class F>
    requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,S&&,F&&> and has_bulk_execute_after_free_function<Ex&&,Ev&&,S&&,F&&>)
  constexpr auto operator()(Ex&& executor, Ev&& event, S&& grid_shape, F&& function) const
  {
    return bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(event), std::forward<S>(grid_shape), std::forward<F>(function));
  }

  // this default path maps coordinates a request for bulk execution in a 1D coordinate system
  // to a bulk_execute_after in the executor's non-1D native coordinate system
  template<executor Ex, event Ev, std::integral I, std::regular_invocable<I> F>
    requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,I,F&&> and
              !has_bulk_execute_after_free_function<Ex&&,Ev&&,I,F&&> and
              !std::integral<executor_coordinate_t<Ex&&>>)
  constexpr auto operator()(Ex&& executor, Ev&& before, I num_points, F&& function) const
  {
    // ask the executor for its preffered grid shape in its native coordinate system
    executor_coordinate_t<Ex> native_grid_shape = bulk_execution_grid(executor, num_points);

    // call bulk_execute_after again with a native grid shape
    return (*this)(std::forward<Ex>(executor), std::forward<Ev>(before), native_grid_shape, [=](executor_coordinate_t<Ex> native_coord)
    {
      // map the native coordinate to a linear index
      std::size_t i = colexicographic_index(native_coord, native_grid_shape);
      
      // if i is a coordinate that the user asked for, invoke the function
      if(i < num_points)
      {
        std::invoke(function,i);
      }
    });
  }


  template<executor Ex, event Ev, grid_coordinate S, std::regular_invocable<S> F>
    requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,S&&,F&&> and
              !has_bulk_execute_after_free_function<Ex&&,Ev&&,S&&,F&&>)
  auto operator()(const Ex& ex, Ev&& before, S&& grid_shape, F&& function) const
  {
    lattice grid{std::forward<S>(grid_shape)};

    // initialize the result event with the first coordinate in the grid
    auto i = grid.begin();
    auto coord = *i;
    auto result = execute_after(ex, before, [function,coord]
    {
      std::invoke(function, coord);
    });

    // build up the result event by adding a dependent event for each function invocation
    for(++i; i != grid.end(); ++i)
    {
      auto coord = *i;

      auto e = execute_after(ex, before, [function,coord]
      {
        std::invoke(function, coord);
      });

      result = dependent_on(ex, std::move(e));
    }

    return result;
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_bulk_execute_after bulk_execute_after;

} // end anonymous namespace


template<class Ex, class Ev, class S, class F>
using bulk_execute_after_result_t = decltype(ASPERA_NAMESPACE::bulk_execute_after(std::declval<Ex>(), std::declval<Ev>(), std::declval<S>(), std::declval<F>()));


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

