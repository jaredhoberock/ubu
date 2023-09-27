#pragma once

#include "../../../detail/prologue.hpp"

#include <array>
#include <concepts>
#include <limits>

namespace ubu::detail
{


template<std::size_t N, std::integral T>
constexpr T nth_root_floor(T x)
{
  // XXX this implementation is far simpler, but requires linking against libmath
  //return std::pow(x, 1./N);
  static_assert(N > 0, "N must be greater than 0");
  
  if (x == 0) return 0;
  if (N == 1) return x;
  
  T low = 1;
  T high = x;
  T result = 0;
  
  while(low <= high)
  {
    T mid = low + (high - low) / 2;
    
    T product = 1;
    bool overflow = false;
    
    for(std::size_t i = 0; i < N and product <= x; ++i)
    {
      if(product > std::numeric_limits<T>::max() / mid)
      {
        overflow = true;
        break;
      }
      product *= mid;
    }
    
    if(not overflow and product == x)
    {
      return mid;
    }
    else if(overflow or product > x)
    {
      high = mid - 1;
    }
    else
    {
      result = mid;
      low = mid + 1;
    }
  }
  
  return result;
}

template<typename T, std::size_t N>
constexpr double product(const std::array<T,N>& factors)
{
  double result = 1;

  for(auto f : factors)
  {
    result *= f;
  }

  return result;
}

template<std::size_t N, std::integral T>
constexpr std::array<T,N> approximate_factors(T x)
{
  std::array<T,N> factors;
  factors.fill(nth_root_floor<N>(x));

  // increment the first smallest factor until we meet or exceed X
  // do this in floating point to avoid dealing with integer overflow
  std::size_t smallest = 0;
  while(product(factors) < double(x))
  {
    factors[smallest]++;
    smallest++;
    smallest %= factors.size();
  }

  return factors;
}


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

