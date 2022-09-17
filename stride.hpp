#pragma once

#include <concepts>
#include <iterator>
#include <ranges>


template<std::random_access_iterator I, std::signed_integral D = std::iter_difference_t<I>>
class stride_iterator
{
  public:
    using base_iterator_type = I;
    using value_type = std::iter_value_t<I>;
    using reference = std::iter_reference_t<I>;
    using difference_type = D;
    using iterator_category = std::random_access_iterator_tag;

    template<std::random_access_iterator OtherI>
      requires std::constructible_from<base_iterator_type, OtherI>
    constexpr stride_iterator(OtherI iter, difference_type stride)
      : current_position_{iter},
        stride_{stride}
    {}

    // XXX this default constructor satisfies libstdc++'s
    // erroneous implementation of std::weakly_incrementable
    constexpr stride_iterator()
      : stride_iterator{base_iterator_type{}, 1}
    {}

    constexpr difference_type stride() const
    {
      return stride_;
    }

    constexpr reference operator*() const
    {
      return *current_position_;
    }

    constexpr reference operator[](difference_type i) const
    {
      auto tmp = *this;
      tmp += i;
      return *tmp;
    }

    constexpr const base_iterator_type& base() const
    {
      return current_position_;
    }

    // pre-increment
    constexpr stride_iterator& operator++()
    {
      current_position_ += stride();
      return *this;
    }

    // post-increment
    constexpr stride_iterator operator++(int)
    {
      stride_iterator result = *this;
      operator++();
      return result;
    }

    // pre-decrement
    constexpr stride_iterator& operator--()
    {
      current_position_ -= stride();
      return *this;
    }

    // post-decrement
    constexpr stride_iterator operator--(int)
    {
      stride_iterator result = *this;
      operator--();
      return result;
    }

    constexpr stride_iterator& operator+=(difference_type n)
    {
      current_position_ += n * stride();
      return *this;
    }

    constexpr stride_iterator operator+(difference_type n) const
    {
      stride_iterator result = *this;
      result += n;
      return result;
    }

    friend constexpr stride_iterator operator+(difference_type lhs, const stride_iterator& rhs)
    {
      return rhs + lhs;
    }

    constexpr stride_iterator& operator-=(difference_type n)
    {
      current_position_ -= n * stride();
      return *this;
    }

    constexpr stride_iterator operator-(difference_type n) const
    {
      stride_iterator result = *this;
      result -= n;
      return result;
    }

    constexpr auto operator<=>(const stride_iterator& rhs) const
    {
      return current_position_ <=> rhs.current_position_;
    }

    // arithmetic between stride_iterators presumes that both iterators' strides are equal
    difference_type operator-(const stride_iterator& rhs) const
    {
      return (base() - rhs.base()) / stride();
    }

  private:
    base_iterator_type current_position_;
    difference_type stride_;
};

template<std::random_access_iterator I1, std::signed_integral D1, std::random_access_iterator I2, std::signed_integral D2>
  requires std::equality_comparable_with<I1,I2> and std::equality_comparable_with<D1,D2>
constexpr bool operator==(const stride_iterator<I1,D1>& lhs, const stride_iterator<I2,D2>& rhs)
{
  return (lhs.base() == rhs.base()) and (lhs.stride() == rhs.stride());
}


static_assert(std::random_access_iterator<stride_iterator<int*>>);


template<std::random_access_iterator I>
class stride_sentinel
{
  public:
    using base_iterator_type = I;

    template<std::random_access_iterator OtherI>
      requires std::constructible_from<I,OtherI>
    constexpr stride_sentinel(OtherI end)
      : end_{end}
    {}

    // satisfy std::semiregular
    stride_sentinel() = default;

    template<std::random_access_iterator OtherI, std::signed_integral OtherD>
      requires std::totally_ordered_with<I,OtherI>
    constexpr bool operator==(const stride_iterator<OtherI,OtherD>& iter) const
    {
      // the stride_iterator has reached the end when it is equal to or past
      // the end of the range
      return end_ <= iter.base();
    }

    constexpr const base_iterator_type& base() const
    {
      return end_;
    }

    template<std::signed_integral D>
    constexpr D operator-(const stride_iterator<I,D>& rhs) const
    {
      return base() - rhs.base();
    }

    template<std::signed_integral D>
    friend constexpr D operator-(const stride_iterator<I,D>& lhs, const stride_sentinel<I>& rhs)
    {
      return lhs.base()- rhs.base();
    }

  private:
    base_iterator_type end_;
};


static_assert(std::sized_sentinel_for<stride_sentinel<int*>, stride_iterator<int*>>);


template<std::ranges::view V, std::signed_integral D = std::ranges::range_difference_t<V>>
class stride_view : public std::ranges::view_interface<stride_view<V,D>>
{
  public:
    using iterator = stride_iterator<std::ranges::iterator_t<V>, D>;
    using sentinel = stride_sentinel<std::ranges::sentinel_t<V>>;

    constexpr stride_view(iterator begin, sentinel end)
      : begin_{begin}, end_{end}
    {}

    constexpr stride_view(V v, D stride)
      : stride_view{iterator{v.begin(), stride}, sentinel{v.end()}}
    {}

    // XXX this default constructor satisfies libstdc++'s
    // erroneous implementation of std::ranges::view
    stride_view() = default;

    constexpr D stride() const
    {
      return begin().stride();
    }

    constexpr iterator begin() const
    {
      return begin_;
    }

    constexpr sentinel end() const
    {
      return end_;
    }

  private:
    iterator begin_;
    sentinel end_;
};


static_assert(std::ranges::view<stride_view<std::ranges::subrange<int*>>>);


template<std::ranges::random_access_range R, std::signed_integral D>
constexpr stride_view<std::views::all_t<R&&>, D> stride(R&& rng, D stride)
{
  return {std::views::all(std::forward<R>(rng)), stride};
}

