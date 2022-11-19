#pragma once

#include "../detail/prologue.hpp"

#include "detail/tuple_algorithm.hpp"
#include "element.hpp"
#include "rank.hpp"
#include <array>
#include <concepts>
#include <functional>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <utility>


namespace ubu
{
namespace detail
{


template<class T, std::size_t N>
class point_base
{
  private:
    std::array<T,N> elements_;

  protected:
    constexpr point_base()
      : elements_{}
    {}

    template<class... OtherT>
      requires (sizeof...(OtherT) == N) and (... and std::convertible_to<OtherT,T>)
    constexpr point_base(OtherT... args)
      : elements_{static_cast<T>(args)...}
    {}

    constexpr T* data()
    {
      return elements_.data();
    }

    constexpr const T* data() const
    {
      return elements_.data();
    }

    template<std::size_t i>
      requires (i < N)
    constexpr T& element()
    {
      return elements_[i];
    }

    template<std::size_t i>
      requires (i < N)
    constexpr const T& element() const
    {
      return elements_[i];
    }

    template<std::size_t i>
      requires (i < N)
    constexpr T&& element() &&
    {
      return std::move(elements_[i]);
    }
};


// small points get named elements
template<class T>
class point_base<T,1>
{
  public:
    T x;

    constexpr point_base()
      : x{}
    {}

    constexpr point_base(T xx)
      : x{xx}
    {}

  protected:
    constexpr T* data()
    {
      return &x;
    }

    constexpr const T* data() const
    {
      return &x;
    }

    template<std::size_t i>
      requires (i == 0)
    constexpr T& element()
    {
      return x;
    }

    template<std::size_t i>
      requires (i == 0)
    constexpr const T& element() const
    {
      return x;
    }

    template<std::size_t i>
      requires (i == 0)
    constexpr T&& element() &&
    {
      return x;
    }
};


template<class T>
class point_base<T,2>
{
  public:
    T x;
    T y;

    constexpr point_base()
      : x{},y{}
    {}

    constexpr point_base(T xx, T yy)
      : x{xx},y{yy}
    {}

  protected:
    constexpr T* data()
    {
      return &x;
    }

    constexpr const T* data() const
    {
      return &x;
    }

    template<std::size_t i>
      requires (i < 2)
    constexpr T& element()
    {
      if constexpr (i == 0) return x;
      return y;
    }

    template<std::size_t i>
      requires (i < 2)
    constexpr const T& element() const
    {
      if constexpr (i == 0) return x;
      return y;
    }

    template<std::size_t i>
      requires (i < 2)
    constexpr T&& element() &&
    {
      if constexpr (i == 0) return std::move(x);
      return std::move(y);
    }
};


template<class T>
class point_base<T,3>
{
  public:
    T x;
    T y;
    T z;

    constexpr point_base()
      : x{},y{},z{}
    {}

    constexpr point_base(T xx, T yy, T zz)
      : x{xx},y{yy},z{zz}
    {}

  protected:
    constexpr T* data()
    {
      return &x;
    }

    constexpr const T* data() const
    {
      return &x;
    }

    template<std::size_t i>
      requires (i < 3)
    constexpr T& element()
    {
      if constexpr (i == 0) return x;
      else if constexpr (i == 1) return y;
      return z;
    }

    template<std::size_t i>
      requires (i < 3)
    constexpr const T& element() const
    {
      if constexpr (i == 0) return x;
      else if constexpr (i == 1) return y;
      return z;
    }

    template<std::size_t i>
      requires (i < 3)
    constexpr T&& element() &&
    {
      if constexpr (i == 0) return std::move(x);
      else if constexpr (i == 1) std::move(y);
      else return std::move(z);
    }
};


template<class T>
class point_base<T,4>
{
  public:
    T x;
    T y;
    T z;
    T w;

    constexpr point_base()
      : x{},y{},z{},w{}
    {}

    constexpr point_base(T xx, T yy, T zz, T ww)
      : x{xx},y{yy},z{zz},w{ww}
    {}

  protected:
    constexpr T* data()
    {
      return &x;
    }

    constexpr const T* data() const
    {
      return &x;
    }

    template<std::size_t i>
      requires (i < 4)
    constexpr T& element()
    {
      if constexpr (i == 0) return x;
      else if constexpr (i == 1) return y;
      else if constexpr (i == 2) return z;
      return w;
    }

    template<std::size_t i>
      requires (i < 4)
    constexpr const T& element() const
    {
      if constexpr (i == 0) return x;
      else if constexpr (i == 1) return y;
      else if constexpr (i == 2) return z;
      return w;
    }

    template<std::size_t i>
      requires (i < 4)
    constexpr T&& element() &&
    {
      if constexpr (i == 0) return std::move(x);
      else if constexpr (i == 1) std::move(y);
      else if constexpr (i == 2) std::move(z);
      return std::move(w);
    }
};


template<class TupleLike, class Indices>
struct has_homogeneous_tuple_elements;

template<class TupleLike, std::size_t Zero, std::size_t... I>
struct has_homogeneous_tuple_elements<TupleLike, std::index_sequence<Zero,I...>>
{
  using tuple_type = std::remove_cvref_t<TupleLike>;

  constexpr static bool value =
    (... and std::same_as<std::tuple_element_t<Zero,tuple_type>, std::tuple_element_t<I,tuple_type>>)
  ;
};


} // end detail


// XXX might also want to insist that the tuple elements have math operations
template<class T>
concept point_like =
  detail::tuple_like<T> and
  detail::has_homogeneous_tuple_elements<T, std::make_index_sequence<rank_v<T>>>::value
;


template<class T, std::size_t N>
concept point_like_of_rank = 
  point_like<T>
  and (rank_v<T> == N)
;


template<point_like T>
using point_element_t = std::tuple_element_t<0,std::remove_cvref_t<T>>;


template<class T, std::size_t N>
class point : public detail::point_base<T,N>
{
  private:
    using super_t = detail::point_base<T,N>;

  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    static constexpr std::size_t rank = N;

    // default constructor
    point() = default;

    // copy constructor
    point(const point&) = default;

    // variadic constructor
    template<class... OtherT>
      requires (sizeof...(OtherT) == N) and (... and std::convertible_to<OtherT,T>)
    constexpr point(OtherT... args)
      : super_t{static_cast<T>(args)...}
    {}

    // converting constructor
    template<class OtherT>
      requires std::convertible_to<OtherT,T>
    constexpr point(const point<OtherT,N>& other)
      : point{other, std::make_index_sequence<N>{}}
    {}

    // fill constructor
    template<class OtherT>
      requires (N > 1) and std::convertible_to<OtherT,T>
    explicit constexpr point(OtherT val)
      : point{val, std::make_index_sequence<N>{}}
    {}


    // observers

    constexpr pointer data()
    {
      return super_t::data();
    }

    constexpr const_pointer data() const
    {
      return super_t::data();
    }

    constexpr static size_type size()
    {
      return N;
    }

    // tuple-like interface
    template<std::size_t i>
      requires (i < N)
    friend constexpr T& get(point& self)
    {
      return self.template element<i>();
    }

    template<std::size_t i>
      requires (i < N)
    friend constexpr const T& get(const point& self)
    {
      return self.template element<i>();
    }

    template<std::size_t i>
      requires (i < N)
    friend constexpr T&& get(point&& self)
    {
      return std::move(self.template element<i>());
    }


    // iterators

    constexpr iterator begin()
    {
      return data();
    }

    constexpr const_iterator begin() const
    {
      return data();
    }

    constexpr const_iterator cbegin()
    {
      return data();
    }

    constexpr const_iterator cbegin() const
    {
      return data();
    }

    constexpr iterator end()
    {
      return data() + size();
    }

    constexpr const_iterator end() const
    {
      return data() + size();
    }

    constexpr const_iterator cend()
    {
      return data() + size();
    }

    constexpr const_iterator cend() const
    {
      return data() + size();
    }

    constexpr reference operator[](size_type pos)
    {
      return begin()[pos];
    }

    constexpr const_reference operator[](size_type pos) const
    {
      return begin()[pos];
    }

    constexpr reference front()
    {
      return *begin();
    }
    
    constexpr const_reference front() const
    {
      return *begin();
    }
    
    constexpr reference back()
    {
      return operator[](N-1);
    }
    
    constexpr const_reference back() const
    {
      return operator[](N-1);
    }


    // relational operators

    template<point_like_of_rank<N> Other>
    constexpr bool operator==(const Other& rhs) const
    {
      return detail::tuple_equal(*this, rhs);
    }

    template<point_like_of_rank<N> Other>
    constexpr bool operator!=(const Other& rhs) const
    {
      return !(*this == rhs);
    }

    // note that point sorts colexicographically, unlike most types
    template<point_like_of_rank<N> Other>
    constexpr bool operator<(const Other& rhs) const
    {
      return detail::tuple_colexicographical_compare(*this, rhs);
    }

    template<point_like_of_rank<N> Other>
    constexpr bool operator>(const Other& rhs) const
    {
      return rhs < *this;
    }

    template<point_like_of_rank<N> Other>
    constexpr bool operator<=(const Other& rhs) const
    {
      return !(*this > rhs);
    }

    template<point_like_of_rank<N> Other>
    constexpr bool operator>=(const Other& rhs) const
    {
      return !(*this < rhs);
    }

    template<point_like_of_rank<N> Other>
    constexpr std::strong_ordering operator<=>(const Other& rhs) const
    {
      if(*this < rhs) return std::strong_ordering::less;
      else if(*this > rhs) return std::strong_ordering::greater;
      return std::strong_ordering::equal;
    }


    // arithmetic assignment operators
    
    template<point_like_of_rank<N> Other>
    constexpr point& operator+=(const Other& rhs)
    {
      detail::tuple_inplace_transform(std::plus{}, *this, rhs);
      return *this;
    }

    template<point_like_of_rank<N> Other>
    constexpr point& operator-=(const Other& rhs)
    {
      detail::tuple_inplace_transform(std::minus{}, *this, rhs);
      return *this;
    }
    
    template<point_like_of_rank<N> Other>
    constexpr point& operator*=(const Other& rhs)
    {
      detail::tuple_inplace_transform(std::multiplies{}, *this, rhs);
      return *this;
    }

    template<point_like_of_rank<N> Other>
    constexpr point& operator/=(const Other& rhs)
    {
      detail::tuple_inplace_transform(std::divides{}, *this, rhs);
      return *this;
    }

    template<point_like_of_rank<N> Other>
      requires (std::integral<T> and std::integral<point_element_t<Other>>)
    constexpr point& operator%=(const Other& rhs)
    {
      detail::tuple_inplace_transform(std::modulus{}, *this, rhs);
      return *this;
    }


    // multiply by scalar
    template<class Other>
      requires std::is_arithmetic_v<Other>
    constexpr point& operator*=(const Other& rhs)
    {
      return *this *= point{rhs};
    }

    // divide by scalar
    template<class Other>
      requires std::is_arithmetic_v<Other>
    constexpr point& operator/=(const Other& rhs)
    {
      return *this /= point{rhs};
    }


    // arithmetic operators

    template<point_like_of_rank<N> Other>
    constexpr point operator+(const Other& rhs) const
    {
      point result = *this;
      result += rhs;
      return result;
    }

    template<point_like_of_rank<N> Other>
    constexpr point operator-(const Other& rhs) const
    {
      point result = *this;
      result -= rhs;
      return result;
    }

    template<point_like_of_rank<N> Other>
    constexpr point operator*(const Other& rhs) const
    {
      point result = *this;
      result *= rhs;
      return result;
    }

    template<point_like_of_rank<N> Other>
    constexpr point operator/(const Other& rhs) const
    {
      point result = *this;
      result /= rhs;
      return result;
    }

    template<point_like_of_rank<N> Other>
      requires (std::integral<T> and std::integral<point_element_t<Other>>)
    constexpr point operator%(const Other& rhs) const
    {
      point result = *this;
      result %= rhs;
      return result;
    }


    // reductions

    constexpr T product() const
    {
      return detail::tuple_fold(*this, [](const T& partial_product, const T& factor)
      {
        return factor * partial_product;
      });
    }

    constexpr T sum() const
    {
      return detail::tuple_fold(*this, [](const T& partial_sum, const T& summand)
      {
        return summand + partial_sum;
      });
    }


    friend std::ostream& operator<<(std::ostream& os, const point& self)
    {
      return detail::tuple_output(os, self);
    }

  private:
    // point unpacking constructor
    template<class OtherT, std::size_t... Indices>
      requires std::convertible_to<OtherT,T>
    constexpr point(const point<OtherT,N>& other, std::index_sequence<Indices...>)
      : point{other[Indices]...}
    {}

    template<std::size_t, class OtherT>
    constexpr static OtherT identity(OtherT value)
    {
      return value;
    }

    // value replicating constructor
    template<class OtherT, std::size_t... Indices>
      requires std::convertible_to<OtherT,T>
    constexpr point(OtherT value, std::index_sequence<Indices...>)
      : point{identity<Indices>(value)...}
    {}
};


// scalar multiply
template<class T1, class T2, std::size_t N>
  requires (std::is_arithmetic_v<T1> and std::is_arithmetic_v<T2> and N > 0)
constexpr point<T2,N> operator*(T1 scalar, const point<T2,N>& p)
{
  return p * scalar;
}


using int1  = point<int,1>;
using int2  = point<int,2>;
using int3  = point<int,3>;
using int4  = point<int,4>;
using int5  = point<int,5>;
using int6  = point<int,6>;
using int7  = point<int,7>;
using int8  = point<int,8>;
using int9  = point<int,9>;
using int10 = point<int,10>;


using uint1  = point<unsigned int,1>;
using uint2  = point<unsigned int,2>;
using uint3  = point<unsigned int,3>;
using uint4  = point<unsigned int,4>;
using uint5  = point<unsigned int,5>;
using uint6  = point<unsigned int,6>;
using uint7  = point<unsigned int,7>;
using uint8  = point<unsigned int,8>;
using uint9  = point<unsigned int,9>;
using uint10 = point<unsigned int,10>;


using size1  = point<size_t,1>;
using size2  = point<size_t,2>;
using size3  = point<size_t,3>;
using size4  = point<size_t,4>;
using size5  = point<size_t,5>;
using size6  = point<size_t,6>;
using size7  = point<size_t,7>;
using size8  = point<size_t,8>;
using size9  = point<size_t,9>;
using size10 = point<size_t,10>;


using float1  = point<float,1>;
using float2  = point<float,2>;
using float3  = point<float,3>;
using float4  = point<float,4>;
using float5  = point<float,5>;
using float6  = point<float,6>;
using float7  = point<float,7>;
using float8  = point<float,8>;
using float9  = point<float,9>;
using float10 = point<float,10>;


using double1  = point<double,1>;
using double2  = point<double,2>;
using double3  = point<double,3>;
using double4  = point<double,4>;
using double5  = point<double,5>;
using double6  = point<double,6>;
using double7  = point<double,7>;
using double8  = point<double,8>;
using double9  = point<double,9>;
using double10 = point<double,10>;


} // end ubu


namespace std
{

// additional tuple-like interface

template<class T, size_t N>
struct tuple_size<ubu::point<T,N>> : std::integral_constant<size_t,N> {};

template<std::size_t I, class T, size_t N>
struct tuple_element<I,ubu::point<T,N>>
{
  using type = T;
};

} // end std


#include "../detail/epilogue.hpp"

