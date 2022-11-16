#pragma once

#include "../detail/prologue.hpp"

#include "colexicographic_decrement.hpp"
#include "colexicographic_increment.hpp"
#include "colexicographic_index.hpp"
#include "colexicographic_index_to_coordinate.hpp"
#include "coordinate.hpp"
#include "coordinate_sum.hpp"
#include "detail/make_coordinate.hpp"
#include "grid_size.hpp"
#include "rank.hpp"
#include <concepts>
#include <initializer_list>
#include <iterator>


namespace ubu
{


namespace detail
{


template<coordinate T> class colexicographic_iterator;


} // end detail


template<coordinate T>
class lattice
{
  public:
    using value_type = T;
    using reference  = value_type;
    using iterator   = detail::colexicographic_iterator<T>;

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
    
    // variadic constructor
    // creates a new lattice at the origin with the given dimensions
    template<std::integral I1, std::integral... Is>
      requires (std::constructible_from<T, I1, Is...> and sizeof...(Is) == (lattice::number_of_dimensions() - 1))
    constexpr explicit lattice(const I1& size1, const Is&... sizes)
      : lattice{detail::make_coordinate<T>(size1, sizes...)}
    {}

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

    // returns whether or not p is the value of a lattice point
    constexpr bool contains(const T& p) const
    {
      return origin() <= p and p < (origin() + shape());
    }

    // returns the number of lattice points
    constexpr std::integral auto size() const
    {
      return ubu::grid_size(shape());
    }

    // returns whether this lattice contains no points
    constexpr bool empty() const
    {
      return shape() == T{};
    }

    // returns the value of the (i,j,k,...)th lattice point
    constexpr T operator[](const T& idx) const
    {
      return origin() + idx;
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

    // reshape does not move the origin
    template<std::integral I1, std::integral... Is>
      requires std::constructible_from<value_type, I1, Is...>
    constexpr void reshape(const I1& size1, const Is&... sizes)
    {
      reshape(detail::make_coordinate<T>(size1, sizes...));
    }

    constexpr iterator begin() const
    {
      return iterator{*this};
    }

    constexpr iterator end() const
    {
      return iterator{*this, iterator::past_the_end(*this)};
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


template<coordinate T>
class colexicographic_iterator
{
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    constexpr colexicographic_iterator(const lattice<T>& domain, T current)
      : domain_{domain},
        current_{current}
    {}

    constexpr explicit colexicographic_iterator(const lattice<T>& domain)
      : colexicographic_iterator{domain, domain.origin()}
    {}

    constexpr reference operator*() const
    {
      return current_;
    }

    constexpr reference operator[](difference_type n) const
    {
      colexicographic_iterator tmp = *this + n;
      return *tmp;
    }

    constexpr colexicographic_iterator& operator++()
    {
      increment();
      return *this;
    }

    constexpr colexicographic_iterator operator++(int)
    {
      colexicographic_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr colexicographic_iterator& operator--()
    {
      decrement();
      return *this;
    }

    constexpr colexicographic_iterator operator--(int)
    {
      colexicographic_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr colexicographic_iterator operator+(difference_type n) const
    {
      colexicographic_iterator result{*this};
      return result += n;
    }

    constexpr colexicographic_iterator& operator+=(difference_type n)
    {
      advance(n);
      return *this;
    }

    constexpr colexicographic_iterator& operator-=(difference_type n)
    {
      return *this += -n;
    }

    constexpr colexicographic_iterator operator-(difference_type n) const
    {
      colexicographic_iterator result{*this};
      return result -= n;
    }

    constexpr difference_type operator-(const colexicographic_iterator& rhs) const
    {
      return colexicographic_index() - rhs.colexicographic_index();
    }

    constexpr bool operator==(const colexicographic_iterator& rhs) const
    {
      return current_ == rhs.current_;
    }

    constexpr bool operator!=(const colexicographic_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator<(const colexicographic_iterator& rhs) const
    {
      return current_ < rhs.current_;
    }

    constexpr bool operator<=(const colexicographic_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    constexpr bool operator>(const colexicographic_iterator& rhs) const
    {
      return rhs < *this;
    }

    constexpr bool operator>=(const colexicographic_iterator &rhs) const
    {
      return !(rhs > *this);
    }

    constexpr static T past_the_end(const lattice<T>& domain)
    {
      // colexicographic_index_to_coordinate rolls over to zero at i == domain.size(), so find the final coordinate in the shape
      T final_coordinate = colexicographic_index_to_coordinate(domain.size() - 1, domain.shape());

      // unlike colexicographic_index_to_coordinate, colexicographic_increment does not roll over at domain.shape()
      // increment the final coordinate in the shape so that we're past the end
      colexicographic_increment(final_coordinate, domain.shape());

      // offset from the origin
      return coordinate_sum(domain.origin(), final_coordinate);
    }

  private:
    constexpr void increment()
    {
      colexicographic_increment(current_, domain_.origin(), coordinate_sum(domain_.origin(), domain_.shape()));
    }

    constexpr void decrement()
    {
      colexicographic_decrement(current_, domain_.origin(), coordinate_sum(domain_.origin(), domain_.shape()));
    }

    constexpr void advance(difference_type n)
    {
      current_ = coordinate_sum(domain_.origin(), colexicographic_index_to_coordinate(colexicographic_index() + n, domain_.shape()));
    }

    constexpr difference_type colexicographic_index() const
    {
      if(is_past_the_end())
      {
        return domain_.size();
      }

      // subtract the origin from current to get
      // 0-based indices along each axis
      T idx = current_ - domain_.origin();

      return ubu::colexicographic_index(idx, domain_.shape());
    }


    constexpr bool is_past_the_end() const
    {
      return current_ == past_the_end(domain_);
    }

    lattice<T> domain_;
    T current_;
};


} // end detail


} // end ubu


#include "../detail/epilogue.hpp"

