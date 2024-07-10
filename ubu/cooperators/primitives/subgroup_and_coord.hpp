#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/tuples.hpp"
#include "../../tensors/coordinates/concepts/coordinate.hpp"
#include "../concepts/cooperator.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_subgroup_and_coord_member_function = requires(T arg)
{
  { arg.subgroup_and_coord() } -> tuples::pair_like;
  { get<0>(arg.subgroup_and_coord()) } -> cooperator;
  { get<1>(arg.subgroup_and_coord()) } -> coordinate;
};

template<class T>
concept has_subgroup_and_coord_free_function = requires(T arg)
{
  { subgroup_and_coord(arg) } -> tuples::pair_like;
  { get<0>(subgroup_and_coord(arg)) } -> cooperator;
  { get<1>(subgroup_and_coord(arg)) } -> coordinate;
};

struct dispatch_subgroup_and_coord
{
  template<class T>
    requires has_subgroup_and_coord_member_function<T&&>
  constexpr auto operator()(T&& arg) const
  {
    return std::forward<T>(arg).subgroup_and_coord();
  }

  template<class T>
    requires (not has_subgroup_and_coord_member_function<T&&>
              and has_subgroup_and_coord_free_function<T&&>)
  constexpr auto operator()(T&& arg) const
  {
    return subgroup_and_coord(std::forward<T>(arg));
  }
};

} // end detail

constexpr detail::dispatch_subgroup_and_coord subgroup_and_coord;

} // end ubu

#include "../../detail/epilogue.hpp"

