#pragma once

#include "../../detail/prologue.hpp"

#include "../composed_tensor.hpp"
#include "../concepts/composable.hpp"
#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "view_base.hpp"
#include <type_traits>

namespace ubu
{


// composed_view and composed_tensor have a cyclic dependency (via compose) which means
// composed_view needs this declaration of composed_tensor
template<class A, view B>
  requires (std::is_object_v<A> and composable<A,B>)
class composed_tensor;


// the only difference between composed_view and composed_tensor is that composed_view
// requires A to be a view if A is also tensor_like and composed_view is itself a view
template<class A, view B>
  requires (std::is_trivially_copy_constructible_v<A> and composable<A,B> and (view<A> or not tensor_like<A>))
class composed_view : public composed_tensor<A,B>, public view_base
{
  public:
    constexpr composed_view(A a, B b) : composed_tensor<A,B>(a,b) {}
  
    composed_view(const composed_view&) = default;

    // delete member all() inherited from composed_tensor
    // this ensures that ubu::all returns exactly a copy of this
    void all() const = delete;
};


namespace detail
{

// composed_view and compose have a cyclic dependency (via composed_tensor) and can't use each other directly
// define make_composed_view for compose's use as soon as composed_view is available
template<class A, view B>
  requires (std::is_trivially_copy_constructible_v<A> and composable<A,B> and (view<A> or not tensor_like<A>))
constexpr view auto make_composed_view(A a, B b)
{
  return composed_view(a,b);
}

} // end detail

} // end ubu

#include "../../detail/epilogue.hpp"

