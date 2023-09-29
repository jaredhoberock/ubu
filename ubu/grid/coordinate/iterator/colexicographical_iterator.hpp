#pragma once

#include "../../../detail/prologue.hpp"

#include "../coordinate_cast.hpp"
#include "../coordinate_difference.hpp"
#include "../coordinate_sum.hpp"
#include "../lift_coordinate.hpp"
#include "../ones.hpp"
#include "colexicographical_decrement_coordinate.hpp"
#include "colexicographical_distance.hpp"
#include "colexicographical_increment_coordinate.hpp"

namespace ubu
{


template<coordinate T>
class colexicographical_iterator
{
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    constexpr colexicographical_iterator(const T& current, const T& origin, const T& shape)
      : current_{current},
        origin_{origin},
        shape_{shape}
    {}

    constexpr colexicographical_iterator(const T& origin, const T& shape)
      : colexicographical_iterator(origin, origin, shape)
    {}

    constexpr colexicographical_iterator(const colexicographical_iterator&) = default;

    constexpr reference operator*() const
    {
      return current_;
    }

    constexpr reference operator[](difference_type n) const
    {
      colexicographical_iterator tmp = *this + n;
      return *tmp;
    }

    constexpr colexicographical_iterator& operator++()
    {
      increment();
      return *this;
    }

    constexpr colexicographical_iterator operator++(int)
    {
      colexicographical_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr colexicographical_iterator& operator--()
    {
      decrement();
      return *this;
    }

    constexpr colexicographical_iterator operator--(int)
    {
      colexicographical_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr colexicographical_iterator operator+(difference_type n) const
    {
      colexicographical_iterator result{*this};
      return result += n;
    }

    constexpr colexicographical_iterator& operator+=(difference_type n)
    {
      advance(n);
      return *this;
    }

    constexpr colexicographical_iterator& operator-=(difference_type n)
    {
      return *this += -n;
    }

    constexpr colexicographical_iterator operator-(difference_type n) const
    {
      colexicographical_iterator result{*this};
      return result -= n;
    }

    constexpr difference_type operator-(const colexicographical_iterator& rhs) const
    {
      return colexicographical_distance(*rhs, current_, shape_);
    }

    constexpr bool operator==(const colexicographical_iterator& rhs) const
    {
      return current_ == rhs.current_;
    }

    constexpr bool operator!=(const colexicographical_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator<(const colexicographical_iterator& rhs) const
    {
      return current_ < rhs.current_;
    }

    constexpr bool operator<=(const colexicographical_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    constexpr bool operator>(const colexicographical_iterator& rhs) const
    {
      return rhs < *this;
    }

    constexpr bool operator>=(const colexicographical_iterator& rhs) const
    {
      return !(rhs > *this);
    }

    constexpr static T end_value(const T& origin, const T& shape)
    {
      T result = last_value(origin, shape);
      colexicographical_increment_coordinate(result, origin, coordinate_sum(origin, shape));
      return result;
    }

    constexpr static T last_value(const T& origin, const T& shape)
    {
      return coordinate_sum(origin, coordinate_difference(shape, ones<T>));
    }

  private:
    constexpr void increment()
    {
      colexicographical_increment_coordinate(current_, origin_, coordinate_sum(origin_, shape_));
    }

    constexpr void decrement()
    {
      colexicographical_decrement_coordinate(current_, origin_, coordinate_sum(origin_, shape_));
    }

    constexpr void advance(difference_type n)
    {
      // XXX ideally, here we would call a function colexicographical_advance_coordinate
      //     instead of relying on colexicographical_index + lift_coordinate

      // this cast is here because lift_coordinate may not return a T
      // XXX note that lift_coordinate uses a colexicographical algorithm
      current_ = coordinate_sum(origin_, coordinate_cast<T>(lift_coordinate(colexicographical_index() + n, shape_)));
    }

    constexpr difference_type colexicographical_index() const
    {
      return colexicographical_distance(origin_, current_, shape_);
    }

    T current_;
    T origin_;
    T shape_;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

