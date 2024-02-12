#pragma once

#include "../detail/prologue.hpp"

#include "compose.hpp"
#include "coordinate/element.hpp"
#include "domain.hpp"
#include "element_exists.hpp"
#include "grid.hpp"
#include "iterator.hpp"
#include "layout/layout.hpp"
#include "shape/shape.hpp"
#include "slice/slice.hpp"
#include <ranges>

namespace ubu
{
namespace detail
{

// view and compose have a cyclic dependency and can't use each other directly
// declare detail::invoke_compose for view's use
template<class... Args>
constexpr auto invoke_compose(Args&&... args);

} // end detail


// XXX consider requiring that Grid and Layout be trivially relocatable
// i.e., Grid can't be std::vector, but it could be a view of std::vector (e.g. a pointer into std::vector)

template<class Grid, layout_for<Grid> Layout>
class view
{
  public:
    using shape_type = grid_shape_t<Layout>;
    using coordinate_type = grid_coordinate_t<Layout>;

    constexpr view(Grid grid, Layout layout)
      : grid_{grid}, layout_{layout}
    {}

    view(const view&) = default;

    constexpr shape_type shape() const
    {
      return ubu::shape(layout_);
    }

    template<class G_ = Grid, class L_ = Layout>
      requires (not grid<G_> and sized_grid<L_>)
    constexpr auto size() const
    {
      return std::ranges::size(layout_);
    }

    // precondition: element_exists(coord)
    template<coordinate_for<Layout> C>
    constexpr decltype(auto) operator[](const C& coord) const
    {
      return ubu::element(grid_, ubu::element(layout_,coord));
    }

    // precondition: in_domain(layout(), coord)
    template<coordinate_for<Layout> C>
    constexpr bool element_exists(const C& coord) const
    {
      if (not ubu::element_exists(layout_, coord)) return false;

      auto to_coord = ubu::element(layout_,coord);

      // if Grid actually fulfills the requirements of ubu::grid,
      // check the coordinate produced by the layout against grid_
      // otherwise, we assume that the layout always perfectly covers grid_
      if constexpr (ubu::grid<Grid>)
      {
        if (not in_domain(grid_, to_coord)) return false;
        if (not ubu::element_exists(grid_, to_coord)) return false;
      }

      return true;
    }

    constexpr Grid grid() const
    {
      return grid_;
    }

    constexpr Layout layout() const
    {
      return layout_;
    }

    // begin is a template because grid_iterator requires its template
    // parameter to be a complete type
    template<class Self = view>
    constexpr grid_iterator<Self> begin() const
    {
      return {*this};
    }
    
    constexpr grid_sentinel end() const
    {
      return {};
    }

    template<slicer_for<coordinate_type> K>
    constexpr ubu::grid auto slice(const K& katana) const
    {
      return detail::invoke_compose(grid(), ubu::slice(layout(), katana));
    }

  private:
    Grid grid_;
    Layout layout_;
};

namespace detail
{

// view and compose have a cyclic dependency and can't use each other directly
// define detail::make_view as soon as view's definition is available
template<class G, layout_for<G> L>
constexpr auto make_view(G g, L l)
{
  return view<G,L>(g,l);
}


} // end detail


} // end ubu

#include "../detail/epilogue.hpp"

