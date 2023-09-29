#pragma once

#include "../detail/prologue.hpp"

#include "coordinate/compare/is_below.hpp"
#include "coordinate/coordinate.hpp"
#include "coordinate/coordinate_sum.hpp"
#include "coordinate/iterator/colexicographical_iterator.hpp"
#include "coordinate/iterator/lexicographical_iterator.hpp"
#include "coordinate/rank.hpp"
#include "shape/shape_size.hpp"
#include <concepts>
#include <initializer_list>
#include <iterator>


namespace ubu
{


template<coordinate T>
class lattice
{
  public:
    using value_type = T;
    using reference  = value_type;
    using iterator   = colexicographical_iterator<T>;

    // default constructor
    lattice() = default;

    // copy constructor
    lattice(const lattice&) = default;

    // (origin, shape) constructor
    // creates a new lattice of the given shape at the given origin
    constexpr lattice(const T& origin, const T& shape)
      : origin_{origin}, shape_{shape}
    {}

    // shape constructor
    // creates a new lattice at the origin with the given shape
    constexpr explicit lattice(const T& shape)
      : lattice{value_type{}, shape}
    {}

    // returns the number of dimensions spanned by this lattice
    static constexpr std::size_t number_of_dimensions()
    {
      return rank_v<T>;
    }

    // returns the value of the smallest lattice point
    constexpr T origin() const
    {
      return origin_;
    }

    // returns the number of lattice points along each of this lattice's dimensions
    constexpr T shape() const
    {
      return shape_;
    }

    // returns whether or not coord is the value of a lattice point
    constexpr bool contains(const T& coord) const
    {
      return is_below_or_equal(origin(), coord) and
             is_below(coord, coordinate_sum(origin(), shape()));
    }

    // returns the number of lattice points
    constexpr std::integral auto size() const
    {
      return shape_size(shape());
    }

    // returns whether this lattice contains no points
    constexpr bool empty() const
    {
      return shape() == T{};
    }

    // returns the value of the (i,j,k,...)th lattice point
    constexpr T operator[](const T& idx) const
    {
      return coordinate_sum(origin(), idx);
    }

    // returns the value of the ith lattice point in colexicographic order
    template<std::integral I>
      requires (rank_v<T> > 1)
    constexpr T operator[](I idx) const
    {
      return begin()[idx];
    }

    // reshape does not move the origin
    constexpr void reshape(const T& shape)
    {
      shape_ = shape;
    }

    constexpr colexicographical_iterator<T> colex_begin() const
    {
      return {origin(), shape()};
    }

    // XXX a colexicographical_sentinel would be more efficient than returning an iterator
    //     because only the final mode needs to be checked to detect the end of the range
    constexpr colexicographical_iterator<T> colex_end() const
    {
      return {colexicographical_iterator<T>::end_value(origin(), shape()), origin(), shape()};
    }

    constexpr lexicographical_iterator<T> lex_begin() const
    {
      return {origin(), shape()};
    }

    // XXX a lexicographical_sentinel would be more efficient than returning an iterator
    //     because only the final mode needs to be checked to detect the end of the range
    constexpr lexicographical_iterator<T> lex_end() const
    {
      return {origin(), shape()};
    }

    constexpr iterator begin() const
    {
      return colex_begin();
    }

    constexpr iterator end() const
    {
      return colex_end();
    }

    constexpr bool operator==(const lattice& other) const
    {
      return (origin_ == other.origin()) and (shape_ == other.shape());
    }

    constexpr bool operator!=(const lattice& other) const
    {
      return !operator==(other);
    }

  private:
    T origin_;
    T shape_;
};


} // end ubu

#include "../detail/epilogue.hpp"

