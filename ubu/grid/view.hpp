#pragma once

#include "../detail/prologue.hpp"

#include "grid.hpp"
#include "iterator.hpp"
#include "layout/layout.hpp"
#include "shape/shape.hpp"
#include "slice/crop_bottom.hpp"
#include "slice/dice_coordinate.hpp"
#include "slice/slice_and_dice.hpp"

namespace ubu
{

// XXX consider requiring that Grid and Layout be trivially relocatable
// i.e., Grid can't be std::vector, but it could be a view of std::vector (e.g. a pointer into std::vector)

template<class Grid, layout_for<Grid> Layout>
class view
{
  public:
    using shape_type = grid_shape_t<Layout>;

    constexpr view(Grid grid, Layout layout)
      : grid_{grid}, layout_{layout}
    {}

    view(const view&) = default;

    constexpr shape_type shape() const
    {
      return ubu::shape(layout_);
    }

    constexpr decltype(auto) operator[](const shape_type& coord) const
    {
      // XXX consider indexing both grid_ and layout_ via a customization point for a bit more flexibility
      //     (i.e., we could support operator() in addition to operator[])
      //     for example such a choice would allow Layout to be a cute::Layout
      //     and Grid could be any function of Layout's element type
      return grid_[layout_[coord]];
    }

    constexpr Grid grid() const
    {
      return grid_;
    }

    constexpr Layout layout() const
    {
      return layout_;
    }

    // begin and end are templates because grid_iterator requires its template
    // parameter to be a complete type
    template<class Self = view>
    constexpr grid_iterator<Self> begin() const
    {
      return {*this};
    }
    
    template<class Self = view>
    constexpr grid_sentinel<Self> end() const
    {
      return {};
    }

    // XXX this returns some type of view
    template<slicer_for<shape_type> K>
    constexpr ubu::grid auto slice(const K& katana) const
    {
      auto [sliced_layout, diced_layout] = slice_and_dice(layout(), katana);
      auto new_origin = diced_layout[dice_coordinate(katana,katana)];
      return make_view(crop_bottom(grid(), new_origin), sliced_layout);
    }

  private:
    template<class G, layout_for<G> L>
    static constexpr view<G,L> make_view(G g, L l)
    {
      return {g,l};
    }

    Grid grid_;
    Layout layout_;
};


} // end ubu

#include "../detail/epilogue.hpp"

