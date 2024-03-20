#pragma once

#include "../detail/prologue.hpp"

#include "concepts/tensor_like.hpp"
#include "coordinate/concepts/congruent.hpp"
#include "traits/tensor_shape.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

// zip_view and zip have a cyclic dependency and can't use each other directly
// declare detail::make_zip_view for zip's use
template<tensor_like T, tensor_like... Ts>
  requires (... and congruent<tensor_shape_t<T>, tensor_shape_t<Ts>>)
constexpr tensor_like auto make_zip_view(T tensor, Ts... tensors);

template<class T, class... Ts>
concept has_zip_member_function = requires(T tensor, Ts... tensors)
{
  { tensor.zip(tensors...) } -> tensor_like;
};

template<class T, class... Ts>
concept has_zip_free_function = requires(T tensor, Ts... tensors)
{
  { zip(tensor, tensors...) } -> tensor_like;
};

struct dispatch_zip
{
  template<class T, class... Ts>
    requires has_zip_member_function<T&&,Ts&&...>
  constexpr tensor_like auto operator()(T&& tensor, Ts&&... tensors) const
  {
    return std::forward<T>(tensor).zip(std::forward<Ts>(tensors)...);
  }

  template<class T, class... Ts>
    requires (not has_zip_member_function<T&&,Ts&&...>
              and has_zip_free_function<T&&,Ts&&...>)
  constexpr tensor_like auto operator()(T&& tensor, Ts&&... tensors) const
  {
    return zip(std::forward<T>(tensor), std::forward<Ts>(tensors)...);
  }

  template<tensor_like T, tensor_like... Ts>
    requires (not has_zip_member_function<T&&,Ts&&...>
              and not has_zip_free_function<T&&,Ts&&...>
              and (... and congruent<tensor_shape_t<T>, tensor_shape_t<Ts>>))
  constexpr tensor_like auto operator()(T&& tensor, Ts&&... tensors) const
  {
    return detail::make_zip_view(std::forward<T>(tensor), std::forward<Ts>(tensors)...);
  }
};

} // end detail

namespace
{

constexpr detail::dispatch_zip zip;

} // end anonymous namespace

namespace detail
{

// zip_view and zip have a cyclic dependency and can't use each other directly
// define detail::invoke_zip as soon as zip's definition is available
template<class... Args>
constexpr auto invoke_zip(Args&&... args)
{
  return zip(std::forward<Args>(args)...);
}

} // end detail


} // end ubu

#include "../detail/epilogue.hpp"

