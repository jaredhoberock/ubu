#pragma once

#include "../detail/prologue.hpp"

#include "../places/memory/pointers/pointer_like.hpp"
#include "../tensors/coordinates/colexicographical_lift.hpp"
#include "../tensors/coordinates/concepts/congruent.hpp"
#include "../tensors/coordinates/concepts/coordinate.hpp"
#include "../tensors/coordinates/traits/rank.hpp"
#include "../tensors/shapes/shape_size.hpp"
#include "../utilities/integrals/size.hpp"
#include "../utilities/tuples.hpp"
#include "barriers.hpp"
#include "concepts/cooperator.hpp"
#include "primitives/id.hpp"
#include "primitives/size.hpp"
#include "primitives/subgroup.hpp"
#include "workspaces.hpp"
#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>

namespace ubu
{


template<workspace W, coordinate S = int, congruent<S> C = S>
struct basic_cooperator
{
  const C coord;
  const S shape;
  static constexpr std::string_view thread_scope = workspace_thread_scope_v<W>;

  constexpr basic_cooperator(const C& c, const S& s, const W& w = W{})
    : coord{c}, shape{s}, workspace_{w}
  {}

  // cooperators with a concurrent workspace can synchronize with the barrier
  template<concurrent_workspace W_ = W>
  constexpr void synchronize() const
  {
    arrive_and_wait(get_barrier(workspace_));
  }

  // cooperators with a stack workspace can deallocate from the workspace
  template<stack_workspace W_ = W>
  constexpr pointer_like auto coop_alloca(int num_bytes)
  {
    return push_allocation(workspace_, num_bytes);
  }

  // cooperators with a stack workspace can allocate from the workspace
  template<stack_workspace W_ = W>
  constexpr void coop_dealloca(int num_bytes)
  {
    pop_allocation(workspace_, num_bytes);
  }

  // precondition: shape_size(new_shape) == size(self)
  template<coordinate OtherS>
  constexpr basic_cooperator<W, OtherS> reshape(const OtherS& new_shape) const
  {
    auto new_coord = colexicographical_lift(id(*this), new_shape);
    return {new_coord, new_shape, workspace_};
  }

  // this overload of subgroup_and_coord requires at least rank 2
  // returns the pair (child_cooperator, child_group_coord)
  template<class = void>
    requires (rank_v<S> > 1 and hierarchical_workspace<W>)
  constexpr auto subgroup_and_coord() const
  {
    using namespace tuples;

    auto child_shape = drop_last_and_unwrap_single(shape);
    auto child_coord = drop_last_and_unwrap_single(coord);
    auto child_group_coord = last(coord);
    auto local_workspace = get_local_workspace(workspace_);

    return std::pair(make_basic_cooperator(child_coord, child_shape, local_workspace), child_group_coord);
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
    auto new_shape = std::pair(tile_shape, size() / shape_size(tile_shape));
  
    return reshape(new_shape).subgroup_and_coord();
  }

  // this overload of subgroup_and_coord uses the child barrier's size as the tile size
  // it returns the pair (child_cooperator, child_group_coord)
  // the requirements are kind of elaborate:
  //   * requires a rank 1 shape and hierarchical workspace
  //   * the local workspace must also have a sized barrier
  // precondition: size(get_local_barrier(get_local_workspace(workspace_))) evenly divides size(*this)
  template<class W_ = W>
    requires (rank_v<S> == 1
              and hierarchical_workspace<W_>
              and concurrent_workspace<local_workspace_t<W_>>
              and sized_barrier_like<barrier_t<local_workspace_t<W_>>>)
  constexpr auto subgroup_and_coord() const
  {
    return tile(ubu::size(get_barrier(get_local_workspace(workspace_))));
  }

  // XXX WAR circle's problem dispatching ubu::size
  //     this member shouldn't be necessary because the tag_invoke(size, semicooperator)
  //     is supposed to be sufficient to provide a size for basic_cooperator
  constexpr auto size() const
  {
    return shape_size(shape);
  }

  private:
    W workspace_;

    template<ubu::workspace OtherW, coordinate OtherS, coordinate OtherC>
    static constexpr basic_cooperator<OtherW,OtherS,OtherC> make_basic_cooperator(OtherC c, OtherS s, OtherW w)
    {
      return {c, s, w};
    }
};

} // end ubu

#include "../detail/epilogue.hpp"

