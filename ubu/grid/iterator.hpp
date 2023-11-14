#pragma once

#include "../detail/prologue.hpp"

#include "grid.hpp"
#include "coordinate/iterator/colexicographical_iterator.hpp"
#include "shape/shape.hpp"
#include <iterator>

namespace ubu
{


template<grid G>
class grid_iterator
{
  public:
    using base_iterator = colexicographical_iterator<grid_shape_t<G>>;

    using iterator_catetory = typename std::iterator_traits<base_iterator>::iterator_category;
    using value_type = grid_element_t<G>;
    using difference_type = typename std::iterator_traits<base_iterator>::difference_type;
    using pointer = void; // XXX do we need a non-void pointer type?
    using reference = grid_reference_t<G>;

    constexpr grid_iterator(G grid)
      : grid_(grid), base_(shape(grid))
    {}

    constexpr decltype(auto) operator*() const
    {
      return grid_[*base_];
    }

    constexpr decltype(auto) operator[](difference_type n) const
    {
      return grid_[base_[n]];
    }

    constexpr grid_iterator& operator++()
    {
      ++base_;
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
      --base_;
      return *this;
    }

    constexpr grid_iterator operator--(int)
    {
      grid_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr grid_iterator operator+(difference_type n) const
    {
      grid_iterator result{*this};
      return result += n;
    }

    constexpr grid_iterator& operator+=(difference_type n)
    {
      base_ += n;
      return *this;
    }

    constexpr grid_iterator& operator-=(difference_type n)
    {
      return *this += -n;
    }

    constexpr grid_iterator operator-(difference_type n) const
    {
      grid_iterator result{*this};
      return result -= n;
    }

    constexpr bool operator==(const grid_iterator& rhs) const
    {
      return base_ == rhs.base_;
    }

    constexpr bool operator!=(const grid_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator<(const grid_iterator& rhs) const
    {
      return base_ < rhs.base_;
    }

    constexpr bool operator<=(const grid_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    constexpr bool operator>(const grid_iterator& rhs) const
    {
      return rhs < *this;
    }

    constexpr bool operator>=(const grid_iterator& rhs) const
    {
      return !(rhs > *this);
    }

    const G& grid() const
    {
      return grid_;
    }

    const base_iterator& base() const
    {
      return base_;
    }

  private:
    // XXX both grid_ and base_ contain some redundant state (for example, shape)
    //     it would be more efficient to store grid_ and the current coordinate
    //     and simply call the increment/decrement coordinate functions directly
    G grid_;
    base_iterator base_;
};


template<grid G>
class grid_sentinel
{
  public:
    friend constexpr bool operator==(const grid_iterator<G>& i, const grid_sentinel<G>& self)
    {
      using base_iterator = typename grid_iterator<G>::base_iterator;

      return *i.base() == base_iterator::end_value(shape(i.grid()));
    }
};


} // end ubu

#include "../detail/epilogue.hpp"

