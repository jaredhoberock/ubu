#pragma once

#include "../../detail/prologue.hpp"

#include "../../memory/buffer/empty_buffer.hpp"
#include "../../tensor/coordinate/colexicographical_lift.hpp"
#include "../../tensor/coordinate/concepts/coordinate.hpp"
#include "../../tensor/coordinate/detail/tuple_algorithm.hpp"
#include "../../tensor/coordinate/traits/rank.hpp"
#include "../../tensor/shape/shape_size.hpp"
#include "../barrier/arrive_and_wait.hpp"
#include "../barrier/sized_barrier_like.hpp"
#include "../workspace/get_local_workspace.hpp"
#include "../workspace/hierarchical_workspace.hpp"
#include "../workspace/workspace.hpp"
#include "../workspace/workspace_thread_scope.hpp"
#include "concepts/cooperator.hpp"
#include "concepts/hierarchical_cooperator.hpp"
#include "descend.hpp"
#include "id.hpp"
#include "size.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace ubu
{


template<workspace W, coordinate S = int, coordinate C = S>
struct basic_cooperator
{
  const C coord;
  const S shape;
  static constexpr std::string_view thread_scope = workspace_thread_scope_v<W>;

  // this ctor creates a child cooperator (e.g. a warp) from its parent hierarchical_cooperator
  // XXX we should enable the construction of any grandchild of arbitrary depth
  template<hierarchical_cooperator P>
    requires std::same_as<basic_cooperator, child_cooperator_t<P>>
  constexpr explicit basic_cooperator(const P& parent)
    : basic_cooperator(descend(parent))
  {}

  constexpr basic_cooperator(const C& c, const S& s, const W& w = W{})
    : coord{c}, shape{s}, workspace_{w}, stack_counter_{0}
  {}

  // cooperators with a concurrent workspace can synchronize with the barrier
  template<class = void>
    requires concurrent_workspace<W>
  constexpr void synchronize() const
  {
    arrive_and_wait(get_barrier(workspace_));
  }

  // cooperators with non-empty buffer can allocate from the buffer
  template<class = void>
    requires nonempty_buffer_like<buffer_t<W>> 
  constexpr std::byte* coop_alloca(int num_bytes)
  {
    std::byte* result = std::ranges::data(get_buffer(workspace_)) + stack_counter_;
    stack_counter_ += num_bytes;
    return result;
  }

  // cooperators with non-empty buffer can deallocate from the buffer
  template<class = void>
    requires nonempty_buffer_like<buffer_t<W>>
  constexpr void coop_dealloca(int num_bytes)
  {
    stack_counter_ -= num_bytes;
  }

  // precondition: shape_size(new_shape) == size(self)
  template<coordinate OtherS>
  constexpr basic_cooperator<W, OtherS> reshape(const OtherS& new_shape) const
  {
    auto new_coord = colexicographical_lift(id(*this), new_shape);
    return {new_coord, new_shape, workspace_};
  }

  // this overload of descend_with_group_coord requires at least rank 2
  // returns the pair (child_cooperator, child_group_coord)
  template<class = void>
    requires (rank_v<S> > 1 and hierarchical_workspace<W>)
  constexpr auto descend_with_group_coord() const
  {
    using namespace detail;

    auto child_shape = tuple_drop_last_and_unwrap_single(shape);
    auto child_coord = tuple_drop_last_and_unwrap_single(coord);
    auto child_group_coord = tuple_last(coord);
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
    auto new_shape = std::pair(tile_shape, size(*this) / shape_size(tile_shape));
  
    return reshape(new_shape).descend_with_group_coord();
  }

  // this overload of tile uses the child barrier's size as the tile size
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
  constexpr auto descend_with_group_coord() const
  {
    return tile(std::ranges::size(get_barrier(get_local_workspace(workspace_))));
  }

  private:
    const W workspace_;
    int stack_counter_; // XXX we should just try to manipulate the workspace's buffer directly rather than keep this extra state around

    template<ubu::workspace OtherW, coordinate OtherS, coordinate OtherC>
    static constexpr basic_cooperator<OtherW,OtherS,OtherC> make_basic_cooperator(OtherC c, OtherS s, OtherW w)
    {
      return {c, s, w};
    }
};

} // end ubu

#include "../../detail/epilogue.hpp"

