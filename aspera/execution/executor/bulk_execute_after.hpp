#pragma once

#include "../../detail/prologue.hpp"

#include "../../coordinate/colexicographic_index.hpp"
#include "../../coordinate/colexicographic_index_to_grid_coordinate.hpp"
#include "../../coordinate/lattice.hpp"
#include "../../coordinate/point.hpp"
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
class dispatch_bulk_execute_after
{
  public:
    // this dispatch path calls the bulk_execute_after member function
    template<class Ex, class Ev, class S, class F>
      requires has_bulk_execute_after_member_function<Ex&&,Ev&&,S&&,F&&>
    constexpr auto operator()(Ex&& executor, Ev&& event, S&& grid_shape, F&& function) const
    {
      return std::forward<Ex>(executor).bulk_execute_after(std::forward<Ev>(event), std::forward<S>(grid_shape), std::forward<F>(function));
    }

    // the following intN overloads exist to allow the user to pass braced lists as the grid_shape parameter
    // like this: bulk_execute_after(ex, before, {x,y,z}, func)
    template<class Ex, class Ev, class F>
      requires has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int2,F&&>
    constexpr auto operator()(Ex&& executor, Ev&& event, aspera::int2 grid_shape, F&& function) const
    {
      return std::forward<Ex>(executor).bulk_execute_after(std::forward<Ev>(event), grid_shape, std::forward<F>(function));
    }

    template<class Ex, class Ev, class F>
      requires has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int3,F&&>
    constexpr auto operator()(Ex&& executor, Ev&& event, aspera::int3 grid_shape, F&& function) const
    {
      return std::forward<Ex>(executor).bulk_execute_after(std::forward<Ev>(event), grid_shape, std::forward<F>(function));
    }

    template<class Ex, class Ev, class F>
      requires has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int4,F&&>
    constexpr auto operator()(Ex&& executor, Ev&& event, aspera::int4 grid_shape, F&& function) const
    {
      return std::forward<Ex>(executor).bulk_execute_after(std::forward<Ev>(event), grid_shape, std::forward<F>(function));
    }



    // this dispatch path calls the bulk_execute_after free function
    template<class Ex, class Ev, class S, class F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,S&&,F&&> and has_bulk_execute_after_free_function<Ex&&,Ev&&,S&&,F&&>)
    constexpr auto operator()(Ex&& executor, Ev&& event, S&& grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(event), std::forward<S>(grid_shape), std::forward<F>(function));
    }

    // the following intN overloads exist to allow the user to pass braced lists as the grid_shape parameter
    // like this: bulk_execute_after(ex, before, {x,y,z}, func)
    template<class Ex, class Ev, class F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int2,F&&> and has_bulk_execute_after_free_function<Ex&&,Ev&&,aspera::int2,F&&>)
    constexpr auto operator()(Ex&& executor, Ev&& event, aspera::int2 grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(event), grid_shape, std::forward<F>(function));
    }

    template<class Ex, class Ev, class F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int3,F&&> and has_bulk_execute_after_free_function<Ex&&,Ev&&,aspera::int3,F&&>)
    constexpr auto operator()(Ex&& executor, Ev&& event, aspera::int3 grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(event), grid_shape, std::forward<F>(function));
    }

    template<class Ex, class Ev, class F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int4,F&&> and has_bulk_execute_after_free_function<Ex&&,Ev&&,aspera::int4,F&&>)
    constexpr auto operator()(Ex&& executor, Ev&& event, aspera::int4 grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(event), grid_shape, std::forward<F>(function));
    }


  private:
    template<executor Ex, event Ev, grid_coordinate G, std::regular_invocable<G> F>
      requires (!std::same_as<executor_coordinate_t<Ex&&>,G>)
    constexpr auto non_native_bulk_execute_after(Ex&& executor, Ev&& before, G grid_shape, F&& function) const
    {
      // ask the executor for its preffered grid shape in its native coordinate system
      executor_coordinate_t<Ex> native_grid_shape = bulk_execution_grid(executor, grid_shape);

      // create a grid of the user's requested coordinates
      lattice grid(grid_shape);

      // call bulk_execute_after again with a native grid shape
      return (*this)(std::forward<Ex>(executor), std::forward<Ev>(before), native_grid_shape, [=](executor_coordinate_t<Ex> native_coord)
      {
        // map the native coordinate to a linear index
        std::size_t i = colexicographic_index(native_coord, native_grid_shape);
        
        // if coord is one of the coordinates that the user asked for, invoke the function
        if(i < grid.size())
        {
          G coord = colexicographic_index_to_grid_coordinate(i, grid_shape);

          std::invoke(function,coord);
        }
      });
    }

  public:
    // this default path maps a request for bulk execution in a non-native coordinate system
    // to a bulk_execute_after in the executor's native coordinate system
    template<executor Ex, event Ev, grid_coordinate G, std::regular_invocable<G> F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,G,F&&> and
                !has_bulk_execute_after_free_function<Ex&&,Ev&&,G,F&&> and
                !std::same_as<executor_coordinate_t<Ex&&>,G>)
    constexpr auto operator()(Ex&& executor, Ev&& before, G grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(before), grid_shape, std::forward<F>(function));
    }


    // the following intN overloads exist to allow the user to pass braced lists as the grid_shape parameter
    // like this: bulk_execute_after(ex, before, {x,y,z}, func)
    template<executor Ex, event Ev, std::regular_invocable<aspera::int2> F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int2,F&&> and
                !has_bulk_execute_after_free_function<Ex&&,Ev&&,aspera::int2,F&&> and
                !std::same_as<executor_coordinate_t<Ex&&>,aspera::int2>)
    constexpr auto operator()(Ex&& executor, Ev&& before, aspera::int2 grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(before), grid_shape, std::forward<F>(function));
    }

    template<executor Ex, event Ev, std::regular_invocable<aspera::int3> F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int3,F&&> and
                !has_bulk_execute_after_free_function<Ex&&,Ev&&,aspera::int3,F&&> and
                !std::same_as<executor_coordinate_t<Ex&&>,aspera::int3>)
    constexpr auto operator()(Ex&& executor, Ev&& before, aspera::int3 grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(before), grid_shape, std::forward<F>(function));
    }

    template<executor Ex, event Ev, std::regular_invocable<aspera::int4> F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,aspera::int4,F&&> and
                !has_bulk_execute_after_free_function<Ex&&,Ev&&,aspera::int4,F&&> and
                !std::same_as<executor_coordinate_t<Ex&&>,aspera::int4>)
    constexpr auto operator()(Ex&& executor, Ev&& before, aspera::int4 grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<Ex>(executor), std::forward<Ev>(before), grid_shape, std::forward<F>(function));
    }



    // this default adapts an executor without a bulk_execute_after function
    template<executor Ex, event Ev, grid_coordinate G, std::regular_invocable<G> F>
      requires (!has_bulk_execute_after_member_function<Ex&&,Ev&&,G,F&&> and
                !has_bulk_execute_after_free_function<Ex&&,Ev&&,G,F&&>)
    auto operator()(const Ex& ex, Ev&& before, G grid_shape, F&& function) const
    {
      lattice grid(grid_shape);

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

