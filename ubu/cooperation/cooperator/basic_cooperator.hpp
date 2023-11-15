#pragma once

#include "../../detail/prologue.hpp"

#include "../../grid/coordinate/colexicographical_lift.hpp"
#include "../../grid/coordinate/coordinate.hpp"
#include "../../grid/coordinate/detail/tuple_algorithm.hpp"
#include "../../grid/coordinate/rank.hpp"
#include "../../grid/shape/shape_size.hpp"
#include "../../grid/size.hpp"
#include "../barrier/arrive_and_wait.hpp"
#include "../barrier/barrier_like.hpp"
#include "../barrier/get_local_barrier.hpp"
#include "cooperator.hpp"
#include "descend.hpp"
#include "hierarchical_cooperator.hpp"
#include "id.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace ubu
{


template<barrier_like B, coordinate S = int, coordinate C = S>
  requires std::copyable<B>
struct basic_cooperator
{
  const B barrier;
  const S shape;
  const C coord;

  // this ctor creates a child cooperator (e.g. a warp) from its parent hierarchical_cooperator
  // XXX we should enable the construction of any grandchild of arbitrary depth
  template<hierarchical_cooperator P>
    requires std::same_as<basic_cooperator, child_cooperator_t<P>>
  constexpr explicit basic_cooperator(const P& parent)
    : basic_cooperator(descend(parent))
  {}

  constexpr basic_cooperator(const B& b, const S& s, const C& c)
    : barrier{b}, shape{s}, coord{c}
  {}

  constexpr void synchronize() const
  {
    return arrive_and_wait(barrier);
  }

  // precondition: shape_size(new_shape) == size(self)
  template<coordinate OtherS>
  constexpr basic_cooperator<B, OtherS> reshape(const OtherS& new_shape) const
  {
    auto new_coord = colexicographical_lift(id(*this), new_shape);
    return {barrier, new_shape, new_coord};
  }

  // this overload of descend_with_group_coord requires at least rank 2
  // returns the pair (child_cooperator, child_group_coord)
  template<class = void>
    requires (rank_v<S> > 1 and hierarchical_barrier_like<B>)
  constexpr auto descend_with_group_coord() const
  {
    using namespace detail;

    auto child_shape = tuple_unwrap_single(tuple_drop_last(shape));
    auto child_coord = tuple_unwrap_single(tuple_drop_last(coord));
    auto child_group_coord = tuple_last(coord);
    auto local_barrier = get_local_barrier(barrier);

    return std::pair(make_basic_cooperator(local_barrier, child_shape, child_coord), child_group_coord);
  }

  // returns the pair (child_cooperator, child_group_coord)
  // precondition: shape_size(tile_shape) evenly divides size(*this)
  // precondition: reshape_cooperator(self, new_shape) is a sliceable_cooperator
  // requires that S be reshapeable and its result be sliceable
  constexpr auto tile(const C& tile_shape) const
  {
    // the basic idea is that we want to take the leading N-1 dimensions of shape
    // and replace them with tile_shape
    // the final dimension becomes the remainder
    // tile_shape needs to "tile" shape for this to work right
    auto new_shape = std::pair(tile_shape, size(*this) / shape_size(tile_shape));
  
    return reshape(new_shape).descend_with_group_coord();
  }

  // this overload of descend_with_group_coord requires rank 1
  // returns the pair (child_cooperator, child_group_coord)
  // this overload of tile uses the local barrier's size as the tile size
  // precondition: size(get_local_barrier(barrier)) evenly divides size(*this)
  template<class B_ = B>
    requires (rank_v<S> == 1
              and hierarchical_barrier_like<B_>
              and sized_barrier_like<local_barrier_t<B_>>)
  constexpr auto descend_with_group_coord() const
  {
    return tile(std::ranges::size(get_local_barrier(barrier)));
  }

  private:
    template<barrier_like OtherB, coordinate OtherS, coordinate OtherC>
    static constexpr basic_cooperator<OtherB,OtherS,OtherC> make_basic_cooperator(OtherB b, OtherS s, OtherC c)
    {
      return {b, s, c};
    }
};


} // end ubu

#include "../../detail/epilogue.hpp"

