#pragma once

#include "../../detail/prologue.hpp"

#include "../../coordinate/grid_coordinate.hpp"
#include "../../coordinate/grid_size.hpp"
#include "executor.hpp"
#include "executor_event.hpp"
#include <concepts>
#include <cstdint>


namespace ubu
{

namespace detail
{


template<grid_coordinate C>
struct bulk_invocable_archetype
{
  void operator()(C) const;
};


template<class E, class C>
concept has_bulk_execute_after_member_function_customization =
  requires(E ex, executor_event_t<E> before, C grid_shape)
  {
    {ex.bulk_execute_after(before, grid_shape, bulk_invocable_archetype<C>{})} -> std::same_as<executor_event_t<E>>;
  }
;


template<class E, class C>
concept has_bulk_execute_after_free_function_customization =
  requires(E ex, executor_event_t<E> before, C grid_shape)
  {
    {bulk_execute_after(ex, before, grid_shape, bulk_invocable_archetype<C>{})} -> std::same_as<executor_event_t<E>>;
  }
;


template<class E, class C>
concept has_bulk_execute_after_customization =
  executor<E> and 
  (has_bulk_execute_after_member_function_customization<E, C> or has_bulk_execute_after_free_function_customization<E, C>)
;


template<class E, class G>
concept has_bulk_execution_grid_member_function =
  executor<E>
  and requires(E ex, G grid_shape)
  {
    {ex.bulk_execution_grid(grid_shape)} -> grid_coordinate;

    requires has_bulk_execute_after_customization<E, decltype(ex.bulk_execution_grid(grid_shape))>;
  }
;

template<class E, class G>
concept has_bulk_execution_grid_free_function =
  executor<E> and
  requires(E ex, G grid_shape)
  {
    {bulk_execution_grid(ex,grid_shape)} -> grid_coordinate;

    requires has_bulk_execute_after_customization<E, decltype(bulk_execution_grid(ex,grid_shape))>;
  }
;


// this is the type of bulk_execution_grid
struct dispatch_bulk_execution_grid
{
  // this path calls the member function
  template<executor E, grid_coordinate G>
    requires has_bulk_execution_grid_member_function<E&&,G>
  constexpr auto operator()(E&& ex, G grid_shape) const
  {
    return std::forward<E>(ex).bulk_execution_grid(grid_shape);
  }

  // this path calls the free function
  template<executor E, grid_coordinate G>
    requires (!has_bulk_execution_grid_member_function<E&&,G> and
               has_bulk_execution_grid_free_function<E&&,G>)
  constexpr auto operator()(E&& ex, G grid_shape) const
  {
    return bulk_execution_grid(std::forward<E>(ex), grid_shape);
  }

  template<executor E, grid_coordinate G>
    requires (!has_bulk_execution_grid_member_function<E&&,G> and
              !has_bulk_execution_grid_free_function<E&&,G>)
  constexpr auto operator()(E&& ex, G grid_shape) const
  {
    // this path maps grid_shape to 1D and recurses
    std::size_t num_points = grid_size(grid_shape);
    return (*this)(std::forward<E>(ex), num_points);
  }

  // default path for size_t just returns the argument
  template<executor E>
    requires (!has_bulk_execution_grid_member_function<E&&,std::size_t> and
              !has_bulk_execution_grid_free_function<E&&,std::size_t>)
  constexpr std::size_t operator()(E&&, std::size_t n) const
  {
    return n;
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_bulk_execution_grid bulk_execution_grid;

} // end anonymous namespace


} // end ubu


#include "../../detail/epilogue.hpp"

