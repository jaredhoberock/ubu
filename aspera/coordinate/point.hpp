#pragma once

#include "../detail/prologue.hpp"

#include "coordinate.hpp"
#include "element.hpp"
#include "index.hpp"
#include "size.hpp"
#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


template<class T, std::size_t N>
  requires (N > 0)
class point
{
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

    // default constructor
    point() = default;

    // copy constructor
    point(const point&) = default;

    // variadic constructor
    template<class... OtherT>
      requires (sizeof...(OtherT) == N) and (... and std::convertible_to<OtherT,T>)
    constexpr point(OtherT... args)
      : elements_{static_cast<T>(args)...}
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
      return elements_.data();
    }

    constexpr const_pointer data() const
    {
      return elements_.data();
    }

    constexpr static size_type size()
    {
      return N;
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

    template<coordinate_of_size<N> Other>
    constexpr bool operator==(const Other& rhs) const
    {
      return equal(*this, rhs, std::make_index_sequence<N>());
    }

    template<coordinate_of_size<N> Other>
    constexpr bool operator!=(const Other& rhs) const
    {
      return !(*this == rhs);
    }

    template<coordinate_of_size<N> Other>
    bool operator<(const Other& rhs) const
    {
      return lexicographical_compare(std::integral_constant<std::size_t,0>{}, *this, rhs);
    }

    template<coordinate_of_size<N> Other>
    bool operator>(const Other& rhs) const
    {
      return lexicographical_compare(std::integral_constant<std::size_t,0>{}, rhs, *this);
    }

    template<coordinate_of_size<N> Other>
    bool operator<=(const Other& rhs) const
    {
      return !(*this > rhs);
    }

    template<coordinate_of_size<N> Other>
    bool operator>=(const Other& rhs) const
    {
      return !(*this < rhs);
    }
    


    // arithmetic assignment operators
    
    template<coordinate_of_size<N> Other>
    constexpr point& operator+=(const Other& rhs)
    {
      inplace_transform(rhs, std::plus{}, std::make_index_sequence<N>{});
      return *this;
    }

    template<coordinate_of_size<N> Other>
    constexpr point& operator-=(const Other& rhs)
    {
      inplace_transform(rhs, std::minus{}, std::make_index_sequence<N>{});
      return *this;
    }
    
    template<coordinate_of_size<N> Other>
    constexpr point& operator*=(const Other& rhs)
    {
      inplace_transform(rhs, std::multiplies{}, std::make_index_sequence<N>{});
      return *this;
    }

    template<coordinate_of_size<N> Other>
    constexpr point& operator/=(const Other& rhs)
    {
      inplace_transform(rhs, std::divides{}, std::make_index_sequence<N>{});
      return *this;
    }

    template<index_of_size<N> Other>
      requires std::integral<T>
    constexpr point& operator%=(const Other& rhs)
    {
      inplace_transform(rhs, std::modulus{}, std::make_index_sequence<N>{});
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

    template<coordinate_of_size<N> Other>
    constexpr point operator+(const Other& rhs) const
    {
      point result = *this;
      result += rhs;
      return result;
    }

    template<coordinate_of_size<N> Other>
    constexpr point operator-(const Other& rhs) const
    {
      point result = *this;
      result -= rhs;
      return result;
    }

    template<coordinate_of_size<N> Other>
    constexpr point operator*(const Other& rhs) const
    {
      point result = *this;
      result *= rhs;
      return result;
    }

    template<coordinate_of_size<N> Other>
    constexpr point operator/(const Other& rhs) const
    {
      point result = *this;
      result /= rhs;
      return result;
    }

    template<index_of_size<N> Other>
      requires std::integral<T>
    constexpr point operator%(const Other& rhs) const
    {
      point result = *this;
      result %= rhs;
      return result;
    }


    // reductions

    constexpr T product() const
    {
      return product_impl(*this, std::make_index_sequence<N>{});
    }

    constexpr T sum() const
    {
      return sum_impl(*this, std::make_index_sequence<N>{});
    }


    friend std::ostream& operator<<(std::ostream& os, const point& self)
    {
      os << "{";
      self.output_elements(os, ", ", std::make_index_sequence<N>{});
      os << "}";

      return os;
    }

  private:
    std::array<T,N> elements_;


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

    template<coordinate_of_size<N> C1, coordinate_of_size<N> C2, std::size_t... Indices>
    constexpr static bool equal(const C1& lhs, const C2& rhs, std::index_sequence<Indices...>)
    {
      return (... and (element<Indices>(lhs) == element<Indices>(rhs)));
    }

    template<class... Types>
    constexpr static void swallow(Types&&... args) {}

    template<coordinate_of_size<N> C, class BinaryOp, std::size_t... Indices>
    constexpr void inplace_transform(const C& rhs, BinaryOp op, std::index_sequence<Indices...>)
    {
      swallow(element<Indices>(*this) = op(element<Indices>(*this), element<Indices>(rhs))...);
    }

    template<coordinate_of_size<N> C1, coordinate_of_size<N> C2>
    constexpr static bool lexicographical_compare(std::integral_constant<std::size_t,N>, const C1&, const C2&)
    {
      return false;
    }

    template<std::size_t cursor, coordinate_of_size<N> C1, coordinate_of_size<N> C2>
    constexpr static bool lexicographical_compare(std::integral_constant<std::size_t,cursor>, const C1& lhs, const C2& rhs)
    {
      if(element<cursor>(lhs) < element<cursor>(rhs)) return true;

      if(element<cursor>(rhs) < element<cursor>(lhs)) return false;

      return lexicographical_compare(std::integral_constant<std::size_t,cursor+1>{}, lhs, rhs);
    }

    template<class Arg>
    static void output_elements(std::ostream& os, const char*, const Arg& arg)
    {
      os << arg;
    }

    template<class Arg, class... Args>
    static void output_elements(std::ostream& os, const char* delimiter, const Arg& arg1, const Args&... args)
    {
      os << arg1 << delimiter;

      output_elements(os, delimiter, args...);
    }

    template<std::size_t... Indices>
    void output_elements(std::ostream& os, const char* delimiter, std::index_sequence<Indices...>) const
    {
      output_elements(os, delimiter, (*this)[Indices]...);
    }


    // XXX these reduction implementations deserve to be generalized and live somewhere common
    template<class... Args>
    constexpr static T product_impl(const T& arg1, const Args&... args)
    {
      return (arg1 * ... * args);
    }

    template<std::size_t... Indices>
    constexpr static T product_impl(const point& p, std::index_sequence<Indices...>)
    {
      return product_impl(p[Indices]...);
    }


    template<class... Args>
    constexpr static T sum_impl(const T& arg1, const Args&... args)
    {
      return (arg1 + ... + args);
    }

    template<std::size_t... Indices>
    constexpr static T sum_impl(const point& p, std::index_sequence<Indices...>)
    {
      return sum_impl(p[Indices]...);
    }
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


ASPERA_NAMESPACE_CLOSE_BRACE


#include "../detail/epilogue.hpp"

