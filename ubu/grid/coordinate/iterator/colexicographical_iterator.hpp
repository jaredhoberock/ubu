#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/coordinate.hpp"
#include "../concepts/congruent.hpp"
#include "../coordinate_difference.hpp"
#include "../coordinate_sum.hpp"
#include "../ones.hpp"
#include "../zeros.hpp"
#include "colexicographical_advance.hpp"
#include "colexicographical_decrement.hpp"
#include "colexicographical_distance.hpp"
#include "colexicographical_increment.hpp"

namespace ubu
{


template<coordinate C, congruent<C> S = C, congruent<C> O = C>
class colexicographical_iterator
{
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = C;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    constexpr colexicographical_iterator(const C& current, const O& origin, const S& shape)
      : current_{current},
        origin_{origin},
        shape_{shape}
    {}

    constexpr colexicographical_iterator(const O& origin, const S& shape)
      : colexicographical_iterator(origin, origin, shape)
    {}

    // this ctor overload assumes that the user is asking for an iterator pointing
    // to the first coordinate of a grid whose origin is at zeros<O> of the given shape
    constexpr colexicographical_iterator(const S& shape)
      : colexicographical_iterator(zeros<O>, shape)
    {}

    colexicographical_iterator(const colexicographical_iterator&) = default;

    colexicographical_iterator() = default;

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

    friend constexpr colexicographical_iterator operator+(difference_type n, const colexicographical_iterator& self)
    {
      return self + n;
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

    constexpr static colexicographical_iterator end(const O& origin, const S& shape)
    {
      return {end_value(origin, shape), origin, shape};
    }

    // this end overload assumes that the user is asking for an iterator pointing
    // to the end of a grid of the given shape whose origin is at zeros<O>
    constexpr static colexicographical_iterator end(const S& shape)
    {
      return end(zeros<O>, shape);
    }

    constexpr static C end_value(const O& origin, const S& shape)
    {
      C result = last_value(origin, shape);
      colexicographical_increment(result, origin, coordinate_sum(origin, shape));
      return result;
    }

    // this overload of end_value assumes the origin is at zeros<O>
    constexpr static C end_value(const S& shape)
    {
      return end_value(zeros<O>, shape);
    }

    constexpr static C last_value(const O& origin, const S& shape)
    {
      return coordinate_sum(origin, coordinate_difference(shape, ones<C>));
    }

  private:
    constexpr void increment()
    {
      colexicographical_increment(current_, origin_, coordinate_sum(origin_, shape_));
    }

    constexpr void decrement()
    {
      colexicographical_decrement(current_, origin_, coordinate_sum(origin_, shape_));
    }

    constexpr void advance(difference_type n)
    {
      colexicographical_advance(current_, shape_, n);
    }

    C current_;
    O origin_;
    S shape_;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

