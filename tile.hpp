#pragma once

#include "stride.hpp"
#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>
#include <utility>


template<std::ranges::view V, std::signed_integral D = std::ranges::range_difference_t<V>>
class tile_iterator
{
  private:
    template<class T, class N>
    using counted_t = decltype(std::views::counted(std::declval<T>(), std::declval<N>()));

    using base_iterator_type = stride_iterator<std::ranges::iterator_t<V>,D>;
    using base_sentinel_type = stride_sentinel<std::ranges::sentinel_t<V>>;

    constexpr tile_iterator(std::ranges::iterator_t<V> begin, std::ranges::sentinel_t<V> end, D tile_size)
      : current_position_{begin,tile_size},
        end_{end}
    {}

  public:
    using difference_type = D;
    using value_type = counted_t<std::ranges::iterator_t<V>, difference_type>;
    using iterator_category = std::random_access_iterator_tag;

    constexpr tile_iterator(V all, difference_type tile_size)
      : tile_iterator{std::ranges::begin(all), std::ranges::end(all), tile_size}
    {}

    tile_iterator() = default;

    constexpr base_iterator_type base() const
    {
      return current_position_;
    }

    constexpr difference_type tile_size() const
    {
      return current_position_.stride();
    }

    constexpr value_type operator*() const
    {
      auto end_of_current_tile = current_position_;
      ++end_of_current_tile;

      if(end_of_current_tile == end_)
      {
        difference_type size_of_last_tile = end_.base() - current_position_.base();
        return std::views::counted(current_position_.base(), size_of_last_tile);
      }

      return std::views::counted(current_position_.base(), tile_size());
    }

    constexpr auto operator[](difference_type i) const
    {
      auto tmp = *this;
      tmp += i;
      return *tmp;
    }

    constexpr tile_iterator& operator++()
    {
      ++current_position_;
      return *this;
    }

    constexpr tile_iterator operator++(int) const
    {
      tile_iterator result = *this;
      operator++();
      return result;
    }

    constexpr tile_iterator& operator--()
    {
      --current_position_;
      return *this;
    }

    constexpr tile_iterator operator--(int) const
    {
      tile_iterator result = *this;
      operator--();
      return result;
    }

    constexpr tile_iterator& operator+=(difference_type n)
    {
      current_position_ += n;
      return *this;
    }

    constexpr tile_iterator operator+(difference_type n) const
    {
      tile_iterator result = *this;
      result += n;
      return result;
    }

    friend constexpr tile_iterator operator+(difference_type lhs, const tile_iterator& rhs)
    {
      return rhs + lhs;
    }

    constexpr tile_iterator& operator-=(difference_type n)
    {
      current_position_ -= n;
      return *this;
    }

    constexpr tile_iterator operator-(difference_type n) const
    {
      tile_iterator result = *this;
      result -= n;
      return result;
    }

    difference_type operator-(const tile_iterator& rhs) const
    {
      return current_position_ - rhs.current_position_;
    }

    constexpr bool operator==(const tile_iterator& rhs) const
    {
      return current_position_ == rhs.current_position_;
    }

    constexpr auto operator<=>(const tile_iterator& rhs) const
    {
      return current_position_ <=> rhs.current_position_;
    }

    constexpr bool operator<(const tile_iterator& rhs) const
    {
      return current_position_ < rhs.current_position_;
    }

    constexpr bool operator<=(const tile_iterator& rhs) const
    {
      return current_position_ <= rhs.current_position_;
    }

    constexpr bool operator>(const tile_iterator& rhs) const
    {
      return current_position_ > rhs.current_position_;
    }

    constexpr bool operator>=(const tile_iterator& rhs) const
    {
      return current_position_ >= rhs.current_position_;
    }

  private:
    base_iterator_type current_position_;
    base_sentinel_type end_;
};

using test_view = std::ranges::subrange<int*>;
using test_type = tile_iterator<test_view>;
static_assert(std::random_access_iterator<test_type>);


template<std::ranges::view V>
class tile_sentinel
{
  public:
    using base_sentinel_type = stride_sentinel<std::ranges::sentinel_t<V>>;

    constexpr tile_sentinel(base_sentinel_type end)
      : end_{end}
    {}

    constexpr tile_sentinel(V all)
      : tile_sentinel{std::ranges::end(all)}
    {}

    // satisfy std::semiregular
    tile_sentinel() = default;

    constexpr base_sentinel_type base() const
    {
      return end_;
    }

    template<std::signed_integral D>
    constexpr bool operator==(const tile_iterator<V,D>& rhs) const
    {
      return base() == rhs.base();
    }

    template<std::signed_integral D>
    constexpr D operator-(const tile_iterator<V,D>& rhs) const
    {
      return (base() - rhs.base() + rhs.tile_size() - 1) / rhs.tile_size();
    }

    template<std::signed_integral D>
    friend constexpr D operator-(const tile_iterator<V,D>& lhs, const tile_sentinel<V>& rhs) const
    {
      return (lhs.base() - rhs.base() + lhs.tile_size() - 1) / lhs.tile_size();
    }

  private:
    base_sentinel_type end_;
};

static_assert(std::sized_sentinel_for<tile_sentinel<test_view>, tile_iterator<test_view>>);


template<std::ranges::view V, std::signed_integral D = std::ranges::range_difference_t<V>>
class tile_view : public std::ranges::view_interface<tile_view<V,D>>
{
  public:
    using iterator = tile_iterator<V,D>;
    using sentinel = tile_sentinel<V>;

    constexpr tile_view(iterator begin, sentinel end)
      : begin_{begin}, end_{end}
    {}

    constexpr tile_view(V v, D tile_size)
      : tile_view{iterator{v, tile_size}, sentinel{std::ranges::end(v)}}
    {}

    // XXX this default constructor satisfies libstdc++'s
    // erroneous implementation of std::ranges::view
    tile_view() = default;

    constexpr D tile_size() const
    {
      return begin().tile_size();
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


static_assert(std::ranges::view<tile_view<test_view>>);


template<std::ranges::random_access_range R, std::signed_integral D>
constexpr tile_view<std::views::all_t<R&&>, D> tile(R&& rng, D tile_size)
{
  return {std::views::all(std::forward<R>(rng)), tile_size};
}

template<std::ranges::random_access_range R, std::signed_integral D>
constexpr tile_view<std::views::all_t<R&&>,D> tile_evenly(R&& rng, D desired_number_of_tiles, D minimum_tile_size = 1)
{
  D tile_size = (std::ranges::size(rng) + desired_number_of_tiles - 1) / desired_number_of_tiles;
  tile_size = std::max(tile_size, minimum_tile_size);
  return tile(std::forward<R>(rng), tile_size);
}

