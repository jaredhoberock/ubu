#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/comparisons/is_below.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/coordinate_sum.hpp"
#include "../coordinates/iterators/colexicographical_iterator.hpp"
#include "../coordinates/iterators/lexicographical_iterator.hpp"
#include "../coordinates/traits/rank.hpp"
#include "../coordinates/zeros.hpp"
#include "../shapes/shape_size.hpp"
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <ranges>


namespace ubu
{


template<coordinate C, congruent<C> S = C>
class lattice : public std::ranges::view_base
{
  public:
    using coordinate_type = C;
    using iterator = colexicographical_iterator<C,S>;

    // default constructor
    lattice() = default;

    // copy constructor
    lattice(const lattice&) = default;

    // (origin, shape) constructor
    // creates a new lattice of the given shape at the given origin
    constexpr lattice(const C& origin, const S& shape)
      : origin_{origin}, shape_{shape}
    {}

    // shape constructor
    // creates a new lattice with the origin at zeros<C> of the given shape
    constexpr explicit lattice(const S& shape)
      : lattice{zeros<C>, shape}
    {}

    // returns the number of dimensions spanned by this lattice
    static constexpr std::size_t number_of_dimensions()
    {
      return rank_v<C>;
    }

    // returns the value of the smallest lattice point
    constexpr C origin() const
    {
      return origin_;
    }

    // returns the number of lattice points along each of this lattice's dimensions
    constexpr S shape() const
    {
      return shape_;
    }

    // returns whether or not coord is the value of a lattice point
    constexpr bool contains(const C& coord) const
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
      return size() == 0;
    }

    // returns the value of the (i,j,k,...)th lattice point
    constexpr C operator[](const C& idx) const
    {
      return coordinate_sum(origin(), idx);
    }

    // returns the value of the ith lattice point in colexicographic order
    template<std::integral I>
      requires (rank_v<C> > 1)
    constexpr C operator[](I idx) const
    {
      return begin()[idx];
    }

    // reshape does not move the origin
    constexpr void reshape(const S& shape)
    {
      shape_ = shape;
    }

    constexpr colexicographical_iterator<C,S> colex_begin() const
    {
      return {origin(), shape()};
    }

    // XXX a colexicographical_sentinel would be more efficient than returning an iterator
    //     because only the final mode needs to be checked to detect the end of the range
    constexpr colexicographical_iterator<C,S> colex_end() const
    {
      return colexicographical_iterator<C,S>::end(origin(), shape());
    }

    constexpr lexicographical_iterator<C,S> lex_begin() const
    {
      return {origin(), shape()};
    }

    // XXX a lexicographical_sentinel would be more efficient than returning an iterator
    //     because only the final mode needs to be checked to detect the end of the range
    constexpr lexicographical_iterator<C,S> lex_end() const
    {
      return lexicographical_iterator<C,S>::end(origin(), shape());
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
    C origin_;
    S shape_;
};


template<coordinate S>
lattice(const S&) -> lattice<S,S>;


} // end ubu

#include "../../detail/epilogue.hpp"

