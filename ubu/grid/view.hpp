#pragma once

#include "../detail/prologue.hpp"

#include "grid.hpp"
#include "layout/layout.hpp"
#include "shape/shape.hpp"

namespace ubu
{

// XXX consider requiring that Grid and Layout be trivially relocatable
// i.e., Grid can't be std::vector, but it could be a view of std::vector (e.g. a pointer into std::vector)

template<class Grid, ubu::layout_for<Grid> Layout>
class view
{
  public:
    using shape_type = ubu::grid_shape_t<Layout>;

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

  private:
    Grid grid_;
    Layout layout_;
};

} // end ubu

#include "../detail/epilogue.hpp"

