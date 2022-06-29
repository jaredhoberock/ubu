#pragma once

#include "../../detail/prologue.hpp"

#include "address.hpp"
#include "download.hpp"
#include "upload.hpp"
#include <concepts>
#include <type_traits>

namespace ubu
{


template<class D>
concept downloader =
  std::is_nothrow_move_constructible_v<std::remove_cvref_t<D>>
  and std::is_nothrow_destructible_v<std::remove_cvref_t<D>>
  and std::equality_comparable<D>
  and address<typename std::remove_cvref_t<D>::address_type>
  and requires(D d, typename std::remove_cvref_t<D>::address_type from, std::size_t num_bytes, void* to)
  {
    ubu::download(d, from, num_bytes, to);
  }
;


template<class U>
concept uploader =
  std::is_nothrow_move_constructible_v<std::remove_cvref_t<U>>
  and std::is_nothrow_destructible_v<std::remove_cvref_t<U>>
  and std::equality_comparable<U>
  and address<typename std::remove_cvref_t<U>::address_type>
  and requires(U u, const void* from, std::size_t num_bytes, typename std::remove_cvref_t<U>::address_type to)
  {
    ubu::upload(u, from, num_bytes, to);
  }
;


template<class L>
concept loader = uploader<L> and downloader<L>;


} // end ubu

#include "../../detail/epilogue.hpp"

