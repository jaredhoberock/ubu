#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/constant.hpp"
#include "../../utilities/tuples.hpp"
#include "concepts/coordinate.hpp"
#include "concepts/semicoordinate.hpp"
#include "concepts/equal_rank.hpp"
#include "concepts/weakly_congruent.hpp"
#include "coordinate_sum.hpp"
#include "detail/to_integral_like.hpp"
#include "traits/rank.hpp"
#include "traits/zeros.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{


template<semicoordinate C>
constexpr bool all_elements_congruent_impl()
{
  using namespace ubu;

  if constexpr (rank_v<C> < 2)
  {
    return true;
  }
  else
  {
    using e_0 = tuples::element_t<0,C>;

    constexpr auto impl = []<std::size_t...I>(std::index_sequence<0,I...>)
    {
      return (... and congruent<e_0,tuples::element_t<I,C>>);
    };

    return impl(tuples::indices_v<C>);
  }
}

template<class T>
concept all_elements_congruent =
  semicoordinate<T>
  and all_elements_congruent_impl<T>()
;


} // end detail

// coordinate_product is a kind of relaxed integer multiplication operation
//
// when congruent<A,B>, then this function is the inner product between a & b and returns integral_like
// if weakly_congruent<A,B>, then this function recursively maps itself across the elements of B
//
// In the examples below, letters of the alphabet represent integers:
//
// * coordinate_product(a, b) => a*b
// * coordinate_product((),()) => 0
// * coordinate_product((a,b,c), (c,d,e)) => a*c + b*d + c*e
// * coordinate_product(a, (b, c, d)) => (a*b, a*c, a*d)
// * coordinate_product(a, ((b,c), d, (e,f))) => ((a*b,a*c), a*d, (a*e,a*f))
template<coordinate A, coordinate B>
  requires weakly_congruent<A,B>
constexpr coordinate auto coordinate_product(const A& a, const B& b)
{
  if constexpr (nullary_coordinate<A> and nullary_coordinate<B>)
  {
    // both a & b are (), return 0
    return 0_c;
  }
  else if constexpr (unary_coordinate<A>)
  {
    if constexpr (unary_coordinate<B>)
    {
      // both a & b are integral, just multiply
      return detail::to_integral_like(a) * detail::to_integral_like(b);
    }
    else
    {
      // b is a tuple, map coordinate_product across it
      return tuples::zip_with(b, [&](const auto& b_i)
      {
        return coordinate_product(a, b_i);
      });
    }
  }
  else
  {
    // stride & coord are non-empty tuples of the same size; inner product

    auto star = [](const auto& a_i, const auto& b_i)
    {
      return coordinate_product(a_i, b_i);
    };

    auto products = tuples::zip_with(a,b,star);

    // the + operation of the inner product depends on whether the
    // individual products are congruent
    //
    // if they are, then the + operation is sum
    // if they are not, then the + operation is append (i.e., we simply return the tuple of products)

    // if each element of the tuple of products is congruent, sum them
    if constexpr (detail::all_elements_congruent<decltype(products)>)
    {
      // sum the products

      auto plus = [](const auto& prev_sum, const auto& product_i)
      {
        return coordinate_sum(prev_sum, product_i);
      };

      auto init = get<0>(products);

      return tuples::fold_left(tuples::drop_first(products), init, plus);
    }
    else
    {
      // return the tuple of products
      return products;
    }
  }
}


template<coordinate A, coordinate B>
  requires weakly_congruent<A,B>
using coordinate_product_result_t = decltype(coordinate_product(std::declval<A>(),std::declval<B>()));


// this variant of coordinate_product allows control over the width of the reuslt type, R
// R must be congruent to the result returned by the other variant, coordinate_product
template<coordinate R, coordinate A, coordinate B>
  requires (weakly_congruent<A,B> and congruent<R, coordinate_product_result_t<A,B>> and not std::is_reference_v<R>)
constexpr R coordinate_product_r(const A& a, const B& b)
{
  if constexpr (   (nullary_coordinate<A> and nullary_coordinate<B>)
                or (unary_coordinate<A> and unary_coordinate<B>))
  {
    // just cast the result of coordinate_product
    return static_cast<R>(coordinate_product(a,b));
  }
  else if constexpr (unary_coordinate<A>)
  {
    // b is a tuple, map coordinate_product across a
    return tuples::zip_with(b, zeros_v<R>, [&](const auto& b_i, auto r_i)
    {
      return coordinate_product_r<decltype(r_i)>(a, b_i);
    });
  }
  else
  {
    // stride & coord are non-empty tuples of the same size; inner product

    auto star = [](const auto& a_i, const auto& b_i)
    {
      return coordinate_product_r<R>(a_i, b_i);
    };

    auto products = tuples::zip_with(a,b,star);

    // the + operation of the inner product depends on whether the
    // individual products are congruent
    //
    // if they are, then the + operation is sum
    // if they are not, then the + operation is append (i.e., we simply return the tuple of products)

    // if each element of the tuple of products is congruent, sum them
    if constexpr (detail::all_elements_congruent<decltype(products)>)
    {
      // sum the products

      auto plus = [](const auto& prev_sum, const auto& product_i)
      {
        return coordinate_sum(prev_sum, product_i);
      };

      auto init = get<0>(products);

      return tuples::fold_left(tuples::drop_first(products), init, plus);
    }
    else
    {
      // return the tuple of products
      return products;
    }
  }
}


} // end ubu

#include "../../detail/epilogue.hpp"

