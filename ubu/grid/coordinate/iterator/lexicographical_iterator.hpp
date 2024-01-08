#pragma once

#include "../../../detail/prologue.hpp"

#include "../coordinate_difference.hpp"
#include "../coordinate_sum.hpp"
#include "../ones.hpp"
#include "lexicographical_advance.hpp"
#include "lexicographical_decrement.hpp"
#include "lexicographical_increment.hpp"

namespace ubu
{


template<coordinate T>
class lexicographical_iterator
{
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    constexpr lexicographical_iterator(const T& current, const T& origin, const T& shape)
      : current_{current},
        origin_{origin},
        shape_{shape}
    {}

    constexpr lexicographical_iterator(const T& origin, const T& shape)
      : lexicographical_iterator(origin, origin, shape)
    {}

    // this ctor overload assumes that the user is asking for an iterator pointing
    // to the first coordinate of a grid whose origin is at zeros<T> of the given shape
    constexpr lexicographical_iterator(const T& shape)
      : lexicographical_iterator(zeros<T>, shape)
    {}

    constexpr lexicographical_iterator(const lexicographical_iterator&) = default;

    constexpr reference operator*() const
    {
      return current_;
    }

    constexpr reference operator[](difference_type n) const
    {
      lexicographical_iterator tmp = *this + n;
      return *tmp;
    }

    constexpr lexicographical_iterator& operator++()
    {
      increment();
      return *this;
    }

    constexpr lexicographical_iterator operator++(int)
    {
      lexicographical_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr lexicographical_iterator& operator--()
    {
      decrement();
      return *this;
    }

    constexpr lexicographical_iterator operator--(int)
    {
      lexicographical_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr lexicographical_iterator operator+(difference_type n) const
    {
      lexicographical_iterator result{*this};
      return result += n;
    }

    constexpr lexicographical_iterator& operator+=(difference_type n)
    {
      advance(n);
      return *this;
    }

    constexpr lexicographical_iterator& operator-=(difference_type n)
    {
      return *this += -n;
    }

    constexpr lexicographical_iterator operator-(difference_type n) const
    {
      lexicographical_iterator result{*this};
      return result -= n;
    }

    constexpr difference_type operator-(const lexicographical_iterator& rhs) const
    {
      return lexicographical_distance(*rhs, current_, shape_);
    }

    constexpr bool operator==(const lexicographical_iterator& rhs) const
    {
      return current_ == rhs.current_;
    }

    constexpr bool operator!=(const lexicographical_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator<(const lexicographical_iterator& rhs) const
    {
      return current_ < rhs.current_;
    }

    constexpr bool operator<=(const lexicographical_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    constexpr bool operator>(const lexicographical_iterator& rhs) const
    {
      return rhs < *this;
    }

    constexpr bool operator>=(const lexicographical_iterator& rhs) const
    {
      return !(rhs > *this);
    }

    constexpr static lexicographical_iterator end(const T& origin, const T& shape)
    {
      return {end_value(origin, shape), origin, shape};
    }

    // this end overload assumes that the user is asking for an iterator pointing
    // to the end of a grid of the given shape whose origin is at zeros<T>
    constexpr static lexicographical_iterator end(const T& shape)
    {
      return {end_value(zeros<T>, shape), zeros<T>, shape};
    }

    constexpr static T end_value(const T& origin, const T& shape)
    {
      T result = last_value(origin, shape);
      lexicographical_increment(result, origin, coordinate_sum(origin, shape));
      return result;
    }

    // this overload of end_value assumes the origin is at zeros<T>
    constexpr static T end_value(const T& shape)
    {
      return end_value(zeros<T>, shape);
    }

    constexpr static T last_value(const T& origin, const T& shape)
    {
      return coordinate_sum(origin, coordinate_difference(shape, ones<T>));
    }

  private:
    constexpr void increment()
    {
      lexicographical_increment(current_, origin_, coordinate_sum(origin_, shape_));
    }

    constexpr void decrement()
    {
      lexicographical_decrement(current_, origin_, coordinate_sum(origin_, shape_));
    }

    constexpr void advance(difference_type n)
    {
      lexicographical_advance(current_, shape_, n);
    }

    T current_;
    T origin_;
    T shape_;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

