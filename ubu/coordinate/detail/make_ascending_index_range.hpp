#pragma once

#include "../../detail/prologue.hpp"

#include <cstdint>
#include <utility>


namespace ubu::detail
{


template<std::size_t Start, typename Indices, std::size_t End>
struct make_ascending_index_range_impl;

template<std::size_t Start, std::size_t... Indices, std::size_t End>
struct make_ascending_index_range_impl<
  Start,
  std::index_sequence<Indices...>, 
  End
>
{
  using type = typename make_ascending_index_range_impl<
    Start + 1,
    std::index_sequence<Indices..., Start>,
    End
  >::type;
};

template<std::size_t End, std::size_t... Indices>
struct make_ascending_index_range_impl<End, std::index_sequence<Indices...>, End>
{
  using type = std::index_sequence<Indices...>;
};

template<std::size_t Begin, std::size_t End>
using make_ascending_index_range = typename make_ascending_index_range_impl<Begin, std::index_sequence<>, End>::type;


} // end ubu::detail

#include "../../detail/epilogue.hpp"

