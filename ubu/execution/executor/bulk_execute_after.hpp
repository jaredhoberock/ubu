#pragma once

#include "../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../grid/coordinate/point.hpp"
#include "../../grid/lattice.hpp"
#include "../../grid/layout/stride/apply_stride.hpp"
#include "../../grid/layout/stride/compact_column_major_stride.hpp"
#include "bulk_execution_grid.hpp"
#include "concepts/executor.hpp"
#include "dependent_on.hpp"
#include "executor_coordinate.hpp"
#include "executor_happening.hpp"
#include "execute_after.hpp"

#include <utility>
#include <vector>


namespace ubu
{

namespace detail
{


template<class E, class H, class S, class F>
concept has_bulk_execute_after_member_function = requires(E executor, H before, S grid_shape, F function) { executor.bulk_execute_after(before, grid_shape, function); };

template<class E, class H, class S, class F>
concept has_bulk_execute_after_free_function = requires(E executor, H before, S grid_shape, F function) { bulk_execute_after(executor, before, grid_shape, function); };


// this is the type of bulk_execute_after
class dispatch_bulk_execute_after
{
  public:
    // this dispatch path calls the bulk_execute_after member function
    template<class E, class H, class S, class F>
      requires has_bulk_execute_after_member_function<E&&,H&&,S&&,F&&>
    constexpr auto operator()(E&& executor, H&& before, S&& grid_shape, F&& function) const
    {
      return std::forward<E>(executor).bulk_execute_after(std::forward<H>(before), std::forward<S>(grid_shape), std::forward<F>(function));
    }

    // the following intN overloads exist to allow the user to pass braced lists as the grid_shape parameter
    // like this: bulk_execute_after(ex, before, {x,y,z}, func)
    template<class E, class H, class F>
      requires has_bulk_execute_after_member_function<E&&,H&&,ubu::int2,F&&>
    constexpr auto operator()(E&& executor, H&& before, ubu::int2 grid_shape, F&& function) const
    {
      return std::forward<E>(executor).bulk_execute_after(std::forward<H>(before), grid_shape, std::forward<F>(function));
    }

    template<class E, class H, class F>
      requires has_bulk_execute_after_member_function<E&&,H&&,ubu::int3,F&&>
    constexpr auto operator()(E&& executor, H&& before, ubu::int3 grid_shape, F&& function) const
    {
      return std::forward<E>(executor).bulk_execute_after(std::forward<H>(before), grid_shape, std::forward<F>(function));
    }

    template<class E, class H, class F>
      requires has_bulk_execute_after_member_function<E&&,H&&,ubu::int4,F&&>
    constexpr auto operator()(E&& executor, H&& before, ubu::int4 grid_shape, F&& function) const
    {
      return std::forward<E>(executor).bulk_execute_after(std::forward<H>(before), grid_shape, std::forward<F>(function));
    }



    // this dispatch path calls the bulk_execute_after free function
    template<class E, class H, class S, class F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,S&&,F&&> and has_bulk_execute_after_free_function<E&&,H&&,S&&,F&&>)
    constexpr auto operator()(E&& executor, H&& before, S&& grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), std::forward<S>(grid_shape), std::forward<F>(function));
    }

    // the following intN overloads exist to allow the user to pass braced lists as the grid_shape parameter
    // like this: bulk_execute_after(ex, before, {x,y,z}, func)
    template<class E, class H, class F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,ubu::int2,F&&> and has_bulk_execute_after_free_function<E&&,H&&,ubu::int2,F&&>)
    constexpr auto operator()(E&& executor, H&& before, ubu::int2 grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), grid_shape, std::forward<F>(function));
    }

    template<class E, class H, class F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,ubu::int3,F&&> and has_bulk_execute_after_free_function<E&&,H&&,ubu::int3,F&&>)
    constexpr auto operator()(E&& executor, H&& before, ubu::int3 grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), grid_shape, std::forward<F>(function));
    }

    template<class E, class H, class F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,ubu::int4,F&&> and has_bulk_execute_after_free_function<E&&,H&&,ubu::int4,F&&>)
    constexpr auto operator()(E&& executor, H&& before, ubu::int4 grid_shape, F&& function) const
    {
      return bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), grid_shape, std::forward<F>(function));
    }


  private:
    template<executor E, happening H, coordinate G, std::regular_invocable<G> F>
      requires (!std::same_as<executor_coordinate_t<E&&>,G>)
    constexpr auto non_native_bulk_execute_after(E&& executor, H&& before, G grid_shape, F&& function) const
    {
      // ask the executor for its preffered grid shape in its native coordinate system
      executor_coordinate_t<E> native_grid_shape = bulk_execution_grid(executor, grid_shape);

      // create a grid of the user's requested coordinates
      lattice grid(grid_shape);

      // call bulk_execute_after again with a native grid shape
      return (*this)(std::forward<E>(executor), std::forward<H>(before), native_grid_shape, [=](executor_coordinate_t<E> native_coord)
      {
        // map the native coordinate to a linear index
        std::size_t i = apply_stride(native_coord, compact_column_major_stride(native_grid_shape));
        
        // if coord is one of the coordinates that the user asked for, invoke the function
        if(i < grid.size())
        {
          G coord = grid[i];

          std::invoke(function,coord);
        }
      });
    }

  public:
    // this default path maps a request for bulk execution in a non-native coordinate system
    // to a bulk_execute_after in the executor's native coordinate system
    template<executor E, happening H, coordinate G, std::regular_invocable<G> F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,G,F&&> and
                !has_bulk_execute_after_free_function<E&&,H&&,G,F&&> and
                !std::same_as<executor_coordinate_t<E&&>,G>)
    constexpr auto operator()(E&& executor, H&& before, G grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), grid_shape, std::forward<F>(function));
    }


    // the following intN overloads exist to allow the user to pass braced lists as the grid_shape parameter
    // like this: bulk_execute_after(ex, before, {x,y,z}, func)
    template<executor E, happening H, std::regular_invocable<ubu::int2> F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,ubu::int2,F&&> and
                !has_bulk_execute_after_free_function<E&&,H&&,ubu::int2,F&&> and
                !std::same_as<executor_coordinate_t<E&&>,ubu::int2>)
    constexpr auto operator()(E&& executor, H&& before, ubu::int2 grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), grid_shape, std::forward<F>(function));
    }

    template<executor E, happening H, std::regular_invocable<ubu::int3> F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,ubu::int3,F&&> and
                !has_bulk_execute_after_free_function<E&&,H&&,ubu::int3,F&&> and
                !std::same_as<executor_coordinate_t<E&&>,ubu::int3>)
    constexpr auto operator()(E&& executor, H&& before, ubu::int3 grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), grid_shape, std::forward<F>(function));
    }

    template<executor E, happening H, std::regular_invocable<ubu::int4> F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,ubu::int4,F&&> and
                !has_bulk_execute_after_free_function<E&&,H&&,ubu::int4,F&&> and
                !std::same_as<executor_coordinate_t<E&&>,ubu::int4>)
    constexpr auto operator()(E&& executor, H&& before, ubu::int4 grid_shape, F&& function) const
    {
      return this->non_native_bulk_execute_after(std::forward<E>(executor), std::forward<H>(before), grid_shape, std::forward<F>(function));
    }



    // this default adapts an executor without a bulk_execute_after function
    template<executor E, happening H, coordinate G, std::regular_invocable<G> F>
      requires (!has_bulk_execute_after_member_function<E&&,H&&,G,F&&> and
                !has_bulk_execute_after_free_function<E&&,H&&,G,F&&>)
    auto operator()(const E& ex, H&& before, G grid_shape, F&& function) const
    {
      lattice grid(grid_shape);

      // initialize the result happening with the first coordinate in the grid
      auto i = grid.begin();
      auto coord = *i;
      auto result = execute_after(ex, before, [function,coord]
      {
        std::invoke(function, coord);
      });

      // build up the result happening by adding a dependent happening for each function invocation
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


template<class E, class H, class S, class F>
using bulk_execute_after_result_t = decltype(ubu::bulk_execute_after(std::declval<E>(), std::declval<H>(), std::declval<S>(), std::declval<F>()));


} // end ubu


#include "../../detail/epilogue.hpp"

