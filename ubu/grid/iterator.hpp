#pragma once

#include "../detail/prologue.hpp"

#include "grid.hpp"
#include "coordinate/iterator/colexicographical_iterator.hpp"
#include "shape/shape.hpp"
#include <iterator>

namespace ubu
{


// this is a sentinel type for any of dense_grid_iterator, sparse_grid_iterator, or grid_iterator
struct grid_sentinel {};


template<dense_grid G>
class dense_grid_iterator
{
  public:
    using coord_iterator = colexicographical_iterator<grid_shape_t<G>>;

    using iterator_category = typename std::iterator_traits<coord_iterator>::iterator_category;
    using value_type = grid_element_t<G>;
    using difference_type = typename std::iterator_traits<coord_iterator>::difference_type;
    using pointer = void; // XXX do we need a non-void pointer type?
    using reference = grid_reference_t<G>;

    constexpr dense_grid_iterator(G grid)
      : grid_(grid), coord_(shape(grid))
    {}

    const G& grid() const
    {
      return grid_;
    }

    const coord_iterator& coord() const
    {
      return coord_;
    }

    constexpr decltype(auto) operator*() const
    {
      return grid_[*coord_];
    }

    constexpr decltype(auto) operator[](difference_type n) const
    {
      return grid_[coord_[n]];
    }

    constexpr dense_grid_iterator& operator++()
    {
      ++coord_;
      return *this;
    }

    constexpr dense_grid_iterator operator++(int)
    {
      dense_grid_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr dense_grid_iterator& operator--()
    {
      --coord_;
      return *this;
    }

    constexpr dense_grid_iterator operator--(int)
    {
      dense_grid_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr dense_grid_iterator& operator+=(difference_type n)
    {
      coord_ += n;
      return *this;
    }

    constexpr dense_grid_iterator operator+(difference_type n) const
    {
      dense_grid_iterator result{*this};
      return result += n;
    }

    constexpr dense_grid_iterator operator-(difference_type n) const
    {
      dense_grid_iterator result{*this};
      return result -= n;
    }

    constexpr bool operator==(const dense_grid_iterator& rhs) const
    {
      return coord_ == rhs.coord_;
    }

    constexpr bool operator!=(const dense_grid_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator<(const dense_grid_iterator& rhs) const
    {
      return coord_ < rhs.coord_;
    }

    constexpr bool operator<=(const dense_grid_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    constexpr bool operator>(const dense_grid_iterator& rhs) const
    {
      return rhs < *this;
    }

    constexpr bool operator>=(const dense_grid_iterator& rhs) const
    {
      return !(rhs > *this);
    }

    constexpr bool operator==(grid_sentinel) const
    {
      // XXX it would be more efficient if the value of coord_iterator::end was state of grid_sentinel
      return coord_ == coord_iterator::end(shape(grid_));
    }

  private:
    // XXX both grid_ and coord_ contain some redundant state (for example, shape)
    //     it would be more efficient to store grid_ and the current coordinate
    //     and simply call the increment/decrement coordinate functions directly
    G grid_;
    coord_iterator coord_;
};


template<ubu::sparse_grid G>
class sparse_grid_iterator
{
  public:
    using coord_iterator = ubu::colexicographical_iterator<ubu::grid_shape_t<G>>;

    using iterator_category = std::forward_iterator_tag; // XXX this could be bidirectional_iterator_tag if we save the beginning of the range
    using value_type        = ubu::grid_element_t<G>;
    using difference_type   = typename std::iterator_traits<coord_iterator>::difference_type;
    using pointer           = void; // XXX do we need a non-void pointer type?
    using reference         = ubu::grid_reference_t<G>;

    constexpr sparse_grid_iterator(G grid)
      : grid_{grid},
        coord_end_{coord_iterator::end(shape(grid))},
        coord_{find_begin(grid, coord_end_)}
    {}

    sparse_grid_iterator(const sparse_grid_iterator&) = default;

    const G& grid() const
    {
      return grid_;
    }

    const coord_iterator& coord() const
    {
      return coord_;
    }

    constexpr decltype(auto) operator*() const
    {
      return grid_[*coord_];
    }

    constexpr sparse_grid_iterator& operator++()
    {
      // find either the first element that exists, or the end of the range
      do
      {
        ++coord_;
      }
      while(coord_ != coord_end_ and not element_exists(grid_, *coord_));

      return *this;
    }

    constexpr sparse_grid_iterator operator++(int)
    {
      sparse_grid_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr bool operator==(const sparse_grid_iterator& rhs) const
    {
      return coord_ == rhs.coord_;
    }

    constexpr bool operator!=(const sparse_grid_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator==(grid_sentinel) const
    {
      return coord_ == coord_end_;
    }
    
  private:
    static constexpr coord_iterator find_begin(G grid, coord_iterator end)
    {
      coord_iterator coord{shape(grid)};

      while(coord != end and not element_exists(grid, *coord))
      {
        ++coord;
      }

      return coord;
    }

    // XXX grid_, coord_end_, and coord_ contain some redundant state (for example, shape)
    //     it would be more efficient to store grid_ and the current coordinate range
    //     and simply call the increment/decrement coordinate functions directly
    G grid_;
    coord_iterator coord_end_;
    coord_iterator coord_;
};


template<grid G>
class grid_iterator;


template<dense_grid G>
class grid_iterator<G> : public dense_grid_iterator<G>
{
  private:
    using super_t = dense_grid_iterator<G>;

  public:
    // iterator traits
    using typename super_t::iterator_category;
    using typename super_t::value_type;
    using typename super_t::difference_type;
    using typename super_t::pointer;
    using typename super_t::reference;

    // ctors, grid, and coord function
    using super_t::super_t;
    using super_t::grid;
    using super_t::coord;

    // iterator interface follows
    using super_t::operator*;
    using super_t::operator[];

    constexpr grid_iterator& operator++()
    {
      super_t::operator++();
      return *this;
    }

    constexpr grid_iterator operator++(int)
    {
      grid_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr grid_iterator& operator--()
    {
      super_t::operator--();
      return *this;
    }

    constexpr grid_iterator operator--(int)
    {
      grid_iterator result = *this;
      --(*this);
      return result;
    }

    grid_iterator& operator+=(difference_type n)
    {
      super_t::operator+=(n);
      return *this;
    }

    constexpr grid_iterator operator+(difference_type n) const
    {
      grid_iterator result{*this};
      return result += n;
    }

    constexpr grid_iterator operator-(difference_type n) const
    {
      grid_iterator result{*this};
      return result -= n;
    }

    using super_t::operator==;
    using super_t::operator!=;
    using super_t::operator<;
    using super_t::operator<=;
    using super_t::operator>;
    using super_t::operator>=;
};


template<sparse_grid G>
class grid_iterator<G> : public sparse_grid_iterator<G>
{
  private:
    using super_t = sparse_grid_iterator<G>;

  public:
    // iterator traits
    using typename super_t::iterator_category;
    using typename super_t::value_type;
    using typename super_t::difference_type;
    using typename super_t::pointer;
    using typename super_t::reference;

    // ctors, grid, and coord function
    using super_t::super_t;
    using super_t::grid;
    using super_t::coord;

    // iterator interface follows
    using super_t::operator*;

    constexpr grid_iterator& operator++()
    {
      super_t::operator++();
      return *this;
    }

    constexpr grid_iterator operator++(int)
    {
      grid_iterator result = *this;
      ++(*this);
      return result;
    }

    using super_t::operator==;
    using super_t::operator!=;
};


} // end ubu

#include "../detail/epilogue.hpp"

