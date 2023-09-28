#pragma once

#include "../detail/prologue.hpp"

#include "coordinate/compare/is_below.hpp"
#include "coordinate/coordinate.hpp"
#include "coordinate/coordinate_cast.hpp"
#include "coordinate/coordinate_difference.hpp"
#include "coordinate/coordinate_sum.hpp"
#include "coordinate/decrement_coordinate.hpp"
#include "coordinate/increment_coordinate.hpp"
#include "coordinate/lift_coordinate.hpp"
#include "coordinate/ones.hpp"
#include "coordinate/rank.hpp"
#include "layout/stride/apply_stride.hpp"
#include "layout/stride/compact_column_major_stride.hpp"
#include "shape/shape_size.hpp"
#include <concepts>
#include <initializer_list>
#include <iterator>


namespace ubu
{
namespace detail
{


template<coordinate T> class lattice_iterator;


} // end detail


template<coordinate T>
class lattice
{
  public:
    using value_type = T;
    using reference  = value_type;
    using iterator   = detail::lattice_iterator<T>;

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

    // returns the value of the ith lattice point in lexicographic order
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

    constexpr iterator begin() const
    {
      return iterator{*this};
    }

    // XXX a lattice_sentinel would be more efficient than returning a lattice_iterator
    //     because only the final mode needs to be checked for equality for detecting
    //     the end of the range
    constexpr iterator end() const
    {
      return {*this, iterator::end_value(*this)};
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


namespace detail
{


// XXX this should be named colexicographical_iterator and it should be organized underneath coordinate/iterator
// XXX we should also have a lexicographical_iterator
//     lattice.begin() should call lattice.colex_begin()
template<coordinate T>
class lattice_iterator
{
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    constexpr lattice_iterator(const lattice<T>& domain, T current)
      : domain_{domain},
        current_{current}
    {}

    constexpr explicit lattice_iterator(const lattice<T>& domain)
      : lattice_iterator{domain, domain.origin()}
    {}

    constexpr reference operator*() const
    {
      return current_;
    }

    constexpr reference operator[](difference_type n) const
    {
      lattice_iterator tmp = *this + n;
      return *tmp;
    }

    constexpr lattice_iterator& operator++()
    {
      increment();
      return *this;
    }

    constexpr lattice_iterator operator++(int)
    {
      lattice_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr lattice_iterator& operator--()
    {
      decrement();
      return *this;
    }

    constexpr lattice_iterator operator--(int)
    {
      lattice_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr lattice_iterator operator+(difference_type n) const
    {
      lattice_iterator result{*this};
      return result += n;
    }

    constexpr lattice_iterator& operator+=(difference_type n)
    {
      advance(n);
      return *this;
    }

    constexpr lattice_iterator& operator-=(difference_type n)
    {
      return *this += -n;
    }

    constexpr lattice_iterator operator-(difference_type n) const
    {
      lattice_iterator result{*this};
      return result -= n;
    }

    constexpr difference_type operator-(const lattice_iterator& rhs) const
    {
      return index() - rhs.index();
    }

    constexpr bool operator==(const lattice_iterator& rhs) const
    {
      return current_ == rhs.current_;
    }

    constexpr bool operator!=(const lattice_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator<(const lattice_iterator& rhs) const
    {
      return current_ < rhs.current_;
    }

    constexpr bool operator<=(const lattice_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    constexpr bool operator>(const lattice_iterator& rhs) const
    {
      return rhs < *this;
    }

    constexpr bool operator>=(const lattice_iterator& rhs) const
    {
      return !(rhs > *this);
    }

    constexpr static T end_value(const lattice<T>& domain)
    {
      T result = last_value(domain);
      increment_coordinate(result, domain.origin(), domain.shape());
      return result;
    }

    constexpr static T last_value(const lattice<T>& domain)
    {
      return coordinate_sum(domain.origin(), coordinate_difference(domain.shape(), ones<T>));
    }

  private:
    constexpr void increment()
    {
      increment_coordinate(current_, domain_.origin(), coordinate_sum(domain_.origin(), domain_.shape()));
    }

    constexpr void decrement()
    {
      decrement_coordinate(current_, domain_.origin(), coordinate_sum(domain_.origin(), domain_.shape()));
    }

    constexpr void advance(difference_type n)
    {
      // this cast is here because lift_coordinate may not return a T
      current_ = coordinate_sum(domain_.origin(), coordinate_cast<T>(lift_coordinate(index() + n, domain_.shape())));
    }

    constexpr difference_type index() const
    {
      if(is_at_the_end())
      {
        return domain_.size();
      }

      // subtract the origin from current to get
      // 0-based indices along each axis
      T coord = coordinate_difference(current_, domain_.origin());

      return apply_stride(coord, compact_column_major_stride(domain_.shape()));
    }


    constexpr bool is_at_the_end() const
    {
      return not domain_.contains(current_);
    }

    lattice<T> domain_;
    T current_;
};


} // end detail
} // end ubu

#include "../detail/epilogue.hpp"

