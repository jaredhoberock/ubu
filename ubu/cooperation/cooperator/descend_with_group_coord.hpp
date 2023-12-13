#pragma once

#include "../../detail/prologue.hpp"

#include "../../grid/coordinate/detail/tuple_algorithm.hpp"
#include "../../grid/coordinate/coordinate.hpp"
#include "concepts/cooperator.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_descend_with_group_coord_member_function = requires(T arg)
{
  { arg.descend_with_group_coord() } -> detail::pair_like;
  { get<0>(arg.descend_with_group_coord()) } -> cooperator;
  { get<1>(arg.descend_with_group_coord()) } -> coordinate;
};

template<class T>
concept has_descend_with_group_coord_free_function = requires(T arg)
{
  { descend_with_group_coord(arg) } -> detail::pair_like;
  { get<0>(descend_with_group_coord(arg)) } -> cooperator;
  { get<1>(descend_with_group_coord(arg)) } -> coordinate;
};

struct dispatch_descend_with_group_coord
{
  template<class T>
    requires has_descend_with_group_coord_member_function<T&&>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).descend_with_group_coord();
  }

  template<class T>
    requires (not has_descend_with_group_coord_member_function<T&&>
              and has_descend_with_group_coord_free_function<T&&>)
  constexpr auto operator()(T&& arg) const
  {
    return descend_with_group_coord(std::forward<T>(arg));
  }
};

} // end detail

// XXX this CPO needs a better name
constexpr detail::dispatch_descend_with_group_coord descend_with_group_coord;

} // end ubu

#include "../../detail/epilogue.hpp"

