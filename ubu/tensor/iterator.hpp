#pragma once

#include "../detail/prologue.hpp"

#include "concepts.hpp"
#include "coordinate/element.hpp"
#include "coordinate/iterator/colexicographical_iterator.hpp"
#include "shape/shape.hpp"
#include "traits.hpp"
#include <concepts>
#include <iterator>

namespace ubu
{


// this is a sentinel type for any of sized_tensor_iterator, unsized_tensor_iterator, or tensor_iterator
struct tensor_sentinel {};


template<sized_tensor_like T>
  requires view<T>
class sized_tensor_iterator
{
  public:
    using coord_iterator = colexicographical_iterator<tensor_coordinate_t<T>, tensor_shape_t<T>>;

    using iterator_category = typename std::iterator_traits<coord_iterator>::iterator_category;
    using value_type = tensor_element_t<T>;
    using difference_type = typename std::iterator_traits<coord_iterator>::difference_type;
    using pointer = void; // XXX do we need a non-void pointer type?
    using reference = tensor_reference_t<T>;

    constexpr sized_tensor_iterator(T tensor)
      : tensor_(tensor), coord_(shape(tensor))
    {}

    sized_tensor_iterator(const sized_tensor_iterator&) = default;

    sized_tensor_iterator() requires std::default_initializable<T> = default;

    constexpr const T& tensor() const
    {
      return tensor_;
    }

    constexpr const coord_iterator& coord() const
    {
      return coord_;
    }

    constexpr decltype(auto) operator*() const
    {
      return element(tensor_, *coord_);
    }

    constexpr decltype(auto) operator[](difference_type n) const
    {
      return element(tensor_, coord_[n]);
    }

    constexpr sized_tensor_iterator& operator++()
    {
      ++coord_;
      return *this;
    }

    constexpr sized_tensor_iterator operator++(int)
    {
      sized_tensor_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr sized_tensor_iterator& operator--()
    {
      --coord_;
      return *this;
    }

    constexpr sized_tensor_iterator operator--(int)
    {
      sized_tensor_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr sized_tensor_iterator& operator+=(difference_type n)
    {
      coord_ += n;
      return *this;
    }

    constexpr sized_tensor_iterator& operator-=(difference_type n)
    {
      return operator+=(-n);
    }

    constexpr sized_tensor_iterator operator+(difference_type n) const
    {
      sized_tensor_iterator result{*this};
      return result += n;
    }

    friend constexpr sized_tensor_iterator operator+(difference_type n, const sized_tensor_iterator& self)
    {
      return self + n;
    }

    constexpr sized_tensor_iterator operator-(difference_type n) const
    {
      sized_tensor_iterator result{*this};
      return result -= n;
    }
    
    constexpr auto operator-(const sized_tensor_iterator& rhs) const
    {
      return coord_ - rhs.coord_;
    }

    constexpr bool operator==(const sized_tensor_iterator& rhs) const
    {
      return coord_ == rhs.coord_;
    }

    constexpr bool operator!=(const sized_tensor_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator<(const sized_tensor_iterator& rhs) const
    {
      return coord_ < rhs.coord_;
    }

    constexpr bool operator<=(const sized_tensor_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    constexpr bool operator>(const sized_tensor_iterator& rhs) const
    {
      return rhs < *this;
    }

    constexpr bool operator>=(const sized_tensor_iterator& rhs) const
    {
      return !(rhs > *this);
    }

    constexpr bool operator==(tensor_sentinel) const
    {
      // XXX it would be more efficient if the value of coord_iterator::end was state of tensor_sentinel
      return coord_ == coord_iterator::end(shape(tensor_));
    }

    friend constexpr bool operator==(const tensor_sentinel& s, const sized_tensor_iterator& self)
    {
      return self == s;
    }

    constexpr bool operator<(tensor_sentinel) const
    {
      // XXX it would be more efficient if the value of coord_iterator::end was state of tensor_sentinel
      return coord_ < coord_iterator::end(shape(tensor_));
    }

    friend constexpr auto operator-(tensor_sentinel, const sized_tensor_iterator& self)
    {
      // XXX it would be more efficient if the value of coord_iterator::end was state of tensor_sentinel
      return coord_iterator::end(shape(self.tensor_)) - self.coord_;
    }

  private:
    // XXX both tensor_ and coord_ contain some redundant state (for example, shape)
    //     it would be more efficient to store tensor_ and the current coordinate
    //     and simply call the increment/decrement coordinate functions directly
    T tensor_;
    coord_iterator coord_;
};


template<tensor_like T>
  requires (not sized_tensor_like<T> and view<T>)
class unsized_tensor_iterator
{
  public:
    using coord_iterator = colexicographical_iterator<tensor_coordinate_t<T>, tensor_shape_t<T>>;

    using iterator_category = std::forward_iterator_tag; // XXX this could be bidirectional_iterator_tag if we save the beginning of the range
    using value_type        = tensor_element_t<T>;
    using difference_type   = typename std::iterator_traits<coord_iterator>::difference_type;
    using pointer           = void; // XXX do we need a non-void pointer type?
    using reference         = tensor_reference_t<T>;

    constexpr unsized_tensor_iterator(T tensor)
      : tensor_{tensor},
        coord_end_{coord_iterator::end(shape(tensor))},
        coord_{find_begin(tensor, coord_end_)}
    {}

    unsized_tensor_iterator(const unsized_tensor_iterator&) = default;

    unsized_tensor_iterator() requires std::default_initializable<T> = default;

    constexpr const T& tensor() const
    {
      return tensor_;
    }

    constexpr const coord_iterator& coord() const
    {
      return coord_;
    }

    constexpr decltype(auto) operator*() const
    {
      return tensor_[*coord_];
    }

    constexpr unsized_tensor_iterator& operator++()
    {
      // find either the first element that exists, or the end of the range
      do
      {
        ++coord_;
      }
      while(coord_ != coord_end_ and not element_exists(tensor_, *coord_));

      return *this;
    }

    constexpr unsized_tensor_iterator operator++(int)
    {
      unsized_tensor_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr bool operator==(const unsized_tensor_iterator& rhs) const
    {
      return coord_ == rhs.coord_;
    }

    constexpr bool operator!=(const unsized_tensor_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    constexpr bool operator==(tensor_sentinel) const
    {
      return coord_ == coord_end_;
    }
    
  private:
    static constexpr coord_iterator find_begin(T tensor, coord_iterator end)
    {
      coord_iterator coord{shape(tensor)};

      while(coord != end and not element_exists(tensor, *coord))
      {
        ++coord;
      }

      return coord;
    }

    // XXX tensor_, coord_end_, and coord_ contain some redundant state (for example, shape)
    //     it would be more efficient to store tensor_ and the current coordinate range
    //     and simply call the increment/decrement coordinate functions directly
    T tensor_;
    coord_iterator coord_end_;
    coord_iterator coord_;
};


template<tensor_like T>
class tensor_iterator;


template<view T>
  requires sized_tensor_like<T>
class tensor_iterator<T> : public sized_tensor_iterator<T>
{
  private:
    using super_t = sized_tensor_iterator<T>;

  public:
    // iterator traits
    using typename super_t::iterator_category;
    using typename super_t::value_type;
    using typename super_t::difference_type;
    using typename super_t::pointer;
    using typename super_t::reference;

    // ctors, tensor, and coord function
    using super_t::super_t;
    using super_t::tensor;
    using super_t::coord;

    // iterator interface follows
    using super_t::operator*;
    using super_t::operator[];

    constexpr tensor_iterator& operator++()
    {
      super_t::operator++();
      return *this;
    }

    constexpr tensor_iterator operator++(int)
    {
      tensor_iterator result = *this;
      ++(*this);
      return result;
    }

    constexpr tensor_iterator& operator--()
    {
      super_t::operator--();
      return *this;
    }

    constexpr tensor_iterator operator--(int)
    {
      tensor_iterator result = *this;
      --(*this);
      return result;
    }

    constexpr tensor_iterator& operator+=(difference_type n)
    {
      super_t::operator+=(n);
      return *this;
    }

    constexpr tensor_iterator& operator-=(difference_type n)
    {
      super_t::operator-=(n);
      return *this;
    }

    constexpr tensor_iterator operator+(difference_type n) const
    {
      tensor_iterator result{*this};
      return result += n;
    }

    friend constexpr tensor_iterator operator+(difference_type n, const tensor_iterator& self)
    {
      return self + n;
    }

    constexpr tensor_iterator operator-(difference_type n) const
    {
      tensor_iterator result{*this};
      return result -= n;
    }

    constexpr auto operator-(const tensor_iterator& rhs) const
    {
      return super_t::operator-(rhs);
    }

    // XXX if we instead try using super_t::operator== for this operator, std::regular complains because of reasons
    bool operator==(const tensor_iterator&) const = default;

    using super_t::operator!=;
    using super_t::operator<;
    using super_t::operator<=;
    using super_t::operator>;
    using super_t::operator>=;
};


template<view T>
  requires (not sized_tensor_like<T>)
class tensor_iterator<T> : public unsized_tensor_iterator<T>
{
  private:
    using super_t = unsized_tensor_iterator<T>;

  public:
    // iterator traits
    using typename super_t::iterator_category;
    using typename super_t::value_type;
    using typename super_t::difference_type;
    using typename super_t::pointer;
    using typename super_t::reference;

    // ctors, tensor, and coord function
    using super_t::super_t;
    using super_t::tensor;
    using super_t::coord;

    // iterator interface follows
    using super_t::operator*;

    constexpr tensor_iterator& operator++()
    {
      super_t::operator++();
      return *this;
    }

    constexpr tensor_iterator operator++(int)
    {
      tensor_iterator result = *this;
      ++(*this);
      return result;
    }

    using super_t::operator==;
    using super_t::operator!=;
};


template<view T>
class enumerated_tensor_iterator : public ubu::tensor_iterator<T>
{
  private:
    using super_t = ubu::tensor_iterator<T>;

  public:
    // ctors
    using super_t::super_t;

    // this iterator returns the pair (tensor_coordinate_t, tensor_reference_t) when dereferenced
    using value_type = std::pair<ubu::tensor_coordinate_t<T>, typename super_t::reference>;

    constexpr value_type operator*() const
    {
      auto coord_iter = super_t::coord();
      return {*coord_iter, super_t::operator*()};
    }

    template<std::random_access_iterator S = super_t>
    constexpr value_type operator[](std::iter_difference_t<S> n) const
    {
      auto coord_iter = super_t::coord();
      return {coord_iter[n], super_t::operator[](n)};
    }

    // XXX we wouldn't have to include the stuff that follows if the base classes used deduced this 

    constexpr enumerated_tensor_iterator& operator++()
    {
      super_t::operator++();
      return *this;
    }

    constexpr enumerated_tensor_iterator operator++(int)
    {
      enumerated_tensor_iterator result = *this;
      ++(*this);
      return result;
    }

    // XXX if we instead try using super_t::operator== for this operator, std::regular complains because of reasons
    bool operator==(const enumerated_tensor_iterator&) const = default;

    using super_t::operator!=;

    // bidirectional_iterator requirements
    constexpr enumerated_tensor_iterator& operator--()
    {
      super_t::operator--();
      return *this;
    }

    constexpr enumerated_tensor_iterator operator--(int)
    {
      enumerated_tensor_iterator result = *this;
      --(*this);
      return result;
    }

    // random_access_iterator requirements
    template<std::random_access_iterator S = super_t>
    constexpr enumerated_tensor_iterator& operator+=(std::iter_difference_t<S> n)
    {
      super_t::operator+=(n);
      return *this;
    }

    template<std::random_access_iterator S = super_t>
    constexpr enumerated_tensor_iterator& operator-=(std::iter_difference_t<S> n)
    {
      super_t::operator-=(n);
      return *this;
    }

    template<std::random_access_iterator S = super_t>
    constexpr enumerated_tensor_iterator operator+(std::iter_difference_t<S> n) const
    {
      enumerated_tensor_iterator result{*this};
      return result += n;
    }

    template<std::random_access_iterator S = super_t>
    friend constexpr enumerated_tensor_iterator operator+(std::iter_difference_t<S> n, const enumerated_tensor_iterator& self)
    {
      return self + n;
    }

    template<std::random_access_iterator S = super_t>
    constexpr enumerated_tensor_iterator operator-(std::iter_difference_t<S> n) const
    {
      enumerated_tensor_iterator result{*this};
      return result -= n;
    }

    template<std::random_access_iterator S = super_t>
    constexpr auto operator-(const enumerated_tensor_iterator& rhs) const
    {
      return super_t::operator-(rhs);
    }
};


// XXX this deduction guide becomes unnecessary after P2582
template<view T>
enumerated_tensor_iterator(T) -> enumerated_tensor_iterator<T>;


} // end ubu

#include "../detail/epilogue.hpp"

