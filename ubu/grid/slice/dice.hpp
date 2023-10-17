#pragma once

#include "../../detail/prologue.hpp"
#include "../grid.hpp"
#include "slicer.hpp"

namespace ubu
{
namespace detail
{

template<class T, class K>
concept has_dice_member_function = requires(T arg, K katana)
{
  { arg.dice(katana) } -> grid;
};

template<class T, class K>
concept has_dice_free_function = requires(T arg, K katana)
{
  { dice(arg,katana) } -> grid;
};

struct dispatch_dice
{
  template<class A, class K>
    requires has_dice_member_function<A&&,K&&>
  constexpr grid auto operator()(A&& arg, K&& katana) const
  {
    return std::forward<A>(arg).dice(std::forward<K>(katana));
  }

  template<class A, class K>
    requires (not has_dice_member_function<A&&,K&&>
              and has_dice_free_function<A&&,K&&>)
  constexpr grid auto operator()(A&& arg, K&& katana) const
  {
    return dice(std::forward<A>(arg), std::forward<K>(katana));
  }

  template<grid A, slicer_for<grid_shape_t<A>> K>
    requires (not has_dice_member_function<A&&,K&&>
              and not has_dice_free_function<A&&,K&&>)
  constexpr void operator()(A&& arg, K&& katana) const
  {
    static_assert(sizeof(A) != 0, "dice(grid) default not yet implemented.");
  }
};

} // end detail

namespace
{

constexpr detail::dispatch_dice dice;

} // end anonymous namespace

} // end ubu

#include "../../detail/epilogue.hpp"

