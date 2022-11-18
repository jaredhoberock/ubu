#pragma once

#include <ubu/coordinate/coordinate.hpp>
#include <ubu/coordinate/coordinate_sum.hpp>
#include <ubu/coordinate/coordinate_to_index.hpp>
#include <ubu/coordinate/decrement_coordinate.hpp>
#include <ubu/coordinate/detail/make_coordinate.hpp>
#include <ubu/coordinate/grid_size.hpp>
#include <ubu/coordinate/increment_coordinate.hpp>
#include <ubu/coordinate/index_to_coordinate.hpp>
#include <ubu/coordinate/rank.hpp>
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
    
    // variadic constructor
    // creates a new lattice at the origin with the given dimensions
    // XXX get rid of this ctor
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
      return iterator{*this, iterator::end_value(*this)};
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

    constexpr bool operator>=(const lattice_iterator &rhs) const
    {
      return !(rhs > *this);
    }

    constexpr static T end_value(const lattice<T>& domain)
    {
      // index_to_coordinate rolls over to zero at i == domain.size(), so find the final coordinate in the shape
      T final_coordinate = index_to_coordinate(domain.size() - 1, domain.shape());

      // unlike index_to_coordinate, increment_coordinate does not roll over at domain.shape()
      // increment the final coordinate in the shape so that we're at the end
      increment_coordinate(final_coordinate, domain.shape());

      // add the offset from the origin
      return coordinate_sum(domain.origin(), final_coordinate);
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
      current_ = coordinate_sum(domain_.origin(), index_to_coordinate(index() + n, domain_.shape()));
    }

    constexpr difference_type index() const
    {
      if(is_at_the_end())
      {
        return domain_.size();
      }

      // subtract the origin from current to get
      // 0-based indices along each axis
      // XXX this needs to be coordinate_difference or something rather than minus
      T coord = current_ - domain_.origin();

      return coordinate_to_index(coord, domain_.shape());
    }


    constexpr bool is_at_the_end() const
    {
      return current_ == end_value(domain_);
    }

    lattice<T> domain_;
    T current_;
};


} // end detail
} // end ubu

