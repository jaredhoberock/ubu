#pragma once

#include "../../detail/prologue.hpp"
#include "../coordinate/coordinate_cast.hpp"
#include "../element_exists.hpp"
#include "../grid.hpp"
#include "../shape/shape.hpp"
#include "../shape/shape_size.hpp"
#include "slice_coordinate.hpp"
#include "slicer.hpp"
#include "unslice_coordinate.hpp"
#include <utility>

namespace ubu
{

template<grid G, slicer_for<grid_shape_t<G>> S>
class slice_view
{
  public:
    constexpr slice_view(G grid, S katana)
      : grid_{grid}, katana_{katana}
    {}

    using shape_type = decltype(slice_coordinate(ubu::shape(std::declval<G>()), std::declval<S>()));

    constexpr shape_type shape() const
    {
      return slice_coordinate(ubu::shape(grid_), katana_);
    }

    constexpr decltype(auto) operator[](shape_type coord) const
    {
      // we have to do a coordinate_cast because unsliced_coord may not return precisely grid_coordinate_t
      // XXX ideally, this kind of cast would happen in a CPO for indexing a grid with a coordinate

      auto grid_coord = coordinate_cast<grid_coordinate_t<G>>(unslice_coordinate(coord, katana_));
      return grid_[grid_coord];
    }

    // XXX providing element_exists when G is sized is a pessimization
    // XXX we only need to provide element_exists if G is sparse,
    //     but std::ranges::size doesn't seem to work correctly
    constexpr bool element_exists(shape_type coord) const
    {
      auto grid_coord = coordinate_cast<grid_coordinate_t<G>>(unslice_coordinate(coord, katana_));
      return ubu::element_exists(grid_, grid_coord);
    }

    // if G is sized, we can provide size
    template<class G_ = G>
      requires sized_grid<G_>
    constexpr bool size() const
    {
      return shape_size(shape());
    }

  private:
    G grid_;
    S katana_;
};

} // end ubu

#include "../../detail/epilogue.hpp"

