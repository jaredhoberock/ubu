#pragma once

#include "../../../detail/prologue.hpp"

#include "../concepts/coordinate.hpp"
#include "../concepts/congruent.hpp"
#include "../coordinate_difference.hpp"
#include "../coordinate_sum.hpp"
#include "../traits/ones.hpp"
#include "../traits/zeros.hpp"
#include "lexicographical_advance.hpp"
#include "lexicographical_decrement.hpp"
#include "lexicographical_distance.hpp"
#include "lexicographical_increment.hpp"

namespace ubu
{


template<coordinate C, congruent<C> S = C, congruent<C> O = C>
class lexicographical_iterator
{
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = C;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    constexpr lexicographical_iterator(const C& current, const O& origin, const S& shape)
      : current_{current},
        origin_{origin},
        shape_{shape}
    {}

    constexpr lexicographical_iterator(const O& origin, const S& shape)
      : lexicographical_iterator(origin, origin, shape)
    {}

    // this ctor overload assumes that the user is asking for an iterator pointing
    // to the first coordinate of a tensor whose origin is at zeros_v<O> of the given shape
    constexpr lexicographical_iterator(const S& shape)
      : lexicographical_iterator(zeros_v<O>, shape)
    {}

    lexicographical_iterator(const lexicographical_iterator&) = default;

    lexicographical_iterator() = default;

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

    friend constexpr lexicographical_iterator operator+(difference_type n, const lexicographical_iterator& self)
    {
      return self + n;
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

    constexpr static lexicographical_iterator end(const O& origin, const S& shape)
    {
      return {end_value(origin, shape), origin, shape};
    }

    // this end overload assumes that the user is asking for an iterator pointing
    // to the end of a tensor of the given shape whose origin is at zeros_v<O>
    constexpr static lexicographical_iterator end(const S& shape)
    {
      return end(zeros_v<O>, shape);
    }

    constexpr static C end_value(const O& origin, const S& shape)
    {
      C result = last_value(origin, shape);
      lexicographical_increment(result, origin, coordinate_sum(origin, shape));
      return result;
    }

    // this overload of end_value assumes the origin is at zeros_v<O>
    constexpr static C end_value(const S& shape)
    {
      return end_value(zeros_v<O>, shape);
    }

    constexpr static C last_value(const O& origin, const S& shape)
    {
      return coordinate_sum(origin, coordinate_difference(shape, ones_v<C>));
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

    C current_;
    O origin_;
    S shape_;
};


} // end ubu

#include "../../../detail/epilogue.hpp"

