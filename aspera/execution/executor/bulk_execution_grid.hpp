#pragma once

#include "../../detail/prologue.hpp"

#include "../../coordinate/grid_coordinate.hpp"
#include "executor.hpp"
#include "executor_event.hpp"
#include <concepts>
#include <cstdint>

ASPERA_NAMESPACE_OPEN_BRACE


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


template<class E>
concept has_bulk_execution_grid_member_function =
  executor<E>
  and requires(E ex, std::size_t n)
  {
    {ex.bulk_execution_grid(n)} -> grid_coordinate;

    requires has_bulk_execute_after_customization<E, decltype(ex.bulk_execution_grid(n))>;
  }
;

template<class E>
concept has_bulk_execution_grid_free_function =
  executor<E> and
  requires(E ex, std::size_t n)
  {
    {bulk_execution_grid(ex,n)} -> grid_coordinate;

    requires has_bulk_execute_after_customization<E, decltype(bulk_execution_grid(ex,n))>;
  }
;


// this is the type of bulk_execution_grid
struct dispatch_bulk_execution_grid
{
  // this path calls the member function
  template<executor E>
    requires has_bulk_execution_grid_member_function<E&&>
  constexpr auto operator()(E&& ex, std::size_t n) const
  {
    return std::forward<E>(ex).bulk_execution_grid(n);
  }

  // this path calls the free function
  template<executor E>
    requires (!has_bulk_execution_grid_member_function<E&&> and
               has_bulk_execution_grid_free_function<E&&>)
  constexpr auto operator()(E&& ex, std::size_t n) const
  {
    return bulk_execution_grid(std::forward<E>(ex), n);
  }

  // default path just returns the argument
  template<executor E>
    requires (!has_bulk_execution_grid_member_function<E&&> and
              !has_bulk_execution_grid_free_function<E&&>)
  constexpr std::size_t operator()(E&& ex, std::size_t n) const
  {
    return n;
  }
};


} // end detail


namespace
{

constexpr detail::dispatch_bulk_execution_grid bulk_execution_grid;

} // end anonymous namespace


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

