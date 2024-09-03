#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../tensors/concepts/tensor.hpp"
#include "../../../tensors/coordinates/comparisons/is_below.hpp"
#include "../../../tensors/coordinates/concepts/bounded_coordinate.hpp"
#include "../../../tensors/coordinates/concepts/congruent.hpp"
#include "../../../tensors/coordinates/concepts/congruent.hpp"
#include "../../../tensors/coordinates/traits/rank.hpp"
#include "../../../tensors/inplace_tensor.hpp"
#include "../../../tensors/shapes/shape.hpp"
#include "../../../tensors/views/all.hpp"
#include "../../../tensors/views/as_rvalue.hpp"
#include "../../../tensors/views/compose.hpp"
#include "../../../tensors/views/inplace.hpp"
#include "../../../tensors/views/layouts/coshape.hpp"
#include "../../../tensors/views/layouts/splitting_layout.hpp"
#include "../../../tensors/views/trim.hpp"
#include "../../../tensors/views/unzip.hpp"
#include "../../../utilities/constant.hpp"
#include "../../causality/after_all.hpp"
#include "../../causality/asynchronous_memory_view_of.hpp"
#include "../../causality/wait.hpp"
#include "../../memory/views/memory_view.hpp"
#include "concepts/asynchronous_allocator.hpp"
#include "concepts/asynchronously_allocatable_with.hpp"
#include "traits/allocator_happening.hpp"
#include "traits/allocator_shape.hpp"
#include "traits/allocator_value.hpp"

#include <array>
#include <concepts>
#include <new>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>


namespace ubu
{


template<asynchronous_allocator A, bounded_coordinate S>
  requires std::is_trivially_copy_constructible_v<A>
class allocator_tensor
{
  private:
    using inner_shape_type = allocator_shape_t<A>;

  public:
    using value_type = allocator_value_t<A>;
    using happening_type = allocator_happening_t<A>;
    using shape_type = coordinate_cat_result_t<inner_shape_type, S>;

    template<tensor OtherAllocs>
      requires congruent<shape_t<OtherAllocs>,S>
    constexpr allocator_tensor(from_tensor_t, OtherAllocs&& allocs)
      : allocs_(from_tensor, all(std::forward<OtherAllocs>(allocs)))
    {}

    template<std::size_t N>
      requires (rank_v<S> == 1)
    constexpr allocator_tensor(std::array<A,N> allocs)
      : allocator_tensor(from_tensor, allocs)
    {}

    template<std::same_as<A>... U>
      requires (rank_v<S> == 1)
    constexpr allocator_tensor(A alloc, U... allocs)
      : allocator_tensor(std::array<A,1+sizeof...(allocs)>{{alloc,allocs...}})
    {}

    allocator_tensor(const allocator_tensor&) = default;

    allocator_tensor() = default;

    template<class T, congruent<shape_type> AS>
    constexpr memory_view_of<T,AS> auto allocate(AS allocation_shape) const
    {
      auto [after, result] = allocate_after<T>(initial_happening(*this), allocation_shape);
      wait(after);
      return result;
    }

    template<class T, happening B, congruent<shape_type> AS>
      requires asynchronously_allocatable_with<T,A,const B&,inner_shape_type>
    constexpr asynchronous_memory_view_of<T,AS> auto allocate_after(const B& before, AS allocation_shape) const
    {
      auto layout = allocation_layout(allocation_shape);

      auto [inner_shape, outer_shape] = coshape(layout);

      // make sure that the request is not too large
      if(not is_below_or_equal(outer_shape, ubu::shape(allocs_)))
      {
        throw std::bad_alloc();
      }

      // we'll map allocate_after over the tensor of allocators
      view auto xfrm = transform(trim(allocs_, outer_shape), [&](auto alloc)
      {
        return ubu::allocate_after<T>(alloc, before, inner_shape);
      });

      // materialize the result of the calls to bulk_allocate_after into a tensor
      inplace_tensor happenings_and_allocations(from_tensor, xfrm);

      // unzip that tensor into a pair of views
      auto [happenings_view, allocations] = unzip(happenings_and_allocations);

      // compose the allocations with our layout
      view auto result_view = compose(inplace(allocations), layout);

      // move the happenings into a tensor
      inplace_tensor happenings(from_tensor, as_rvalue(happenings_view));

      // "reduce" the happenings into a single happening
      auto result_happening = after_all(happenings);

      return std::pair(std::move(result_happening), result_view);
    }

    template<memory_view V, congruent<S> S_ = tensor_shape_t<decompose_result_first_t<V>>>
    constexpr void deallocate(V tensor) const
    {
      wait(deallocate_after(initial_happening(*this), tensor));
    }

    template<happening B, decomposable V>
      requires congruent<S,tensor_shape_t<decompose_result_first_t<V>>>
               and asynchronously_deallocatable_with<A, const B&, tensor_element_t<decompose_result_first_t<V>>>
    constexpr happening auto deallocate_after(const B& before, V tensor) const
    {
      using namespace ubu;

      // get the underlying allocations
      auto [tensors,_] = decompose(tensor);

      // map deallocate_after across the tensors
      view auto xfrm = transform(tensors, allocs_, [&](auto t, auto a)
      {
        return ubu::deallocate_after(a, before, t);
      });

      // materialize the deallocate calls into a tensor of postcedents
      inplace_tensor postcedents(from_tensor, xfrm);

      // "reduce" the postcedents into a single postcedent
      return after_all(postcedents);
    }

    constexpr bool operator==(const allocator_tensor& other) const
    {
      return std::ranges::equal(allocs_, other.allocs_);
    }

    constexpr bool operator!=(const allocator_tensor& other) const
    {
      return not (*this == other);
    }

  private:
    template<congruent<shape_type> AS>
    static constexpr layout auto allocation_layout(AS allocation_shape)
    {
      // the layout simply splits coordinates of the allocation into
      // a pair (inner_coord, outer_coord)
      // so the split point is at rank_v<inner_shape_type>

      return splitting_layout<rank_v<inner_shape_type>,AS>(allocation_shape);
    }

    inplace_tensor<A,S> allocs_;
};

template<tensor Allocs>
  requires shaped_and_bounded<Allocs>
allocator_tensor(from_tensor_t, Allocs&&) -> allocator_tensor<tensor_element_t<Allocs&&>, tensor_shape_t<Allocs&&>>;

template<asynchronous_allocator A, std::same_as<A>... Allocs>
allocator_tensor(A alloc, Allocs... allocs) -> allocator_tensor<A, constant<1+sizeof...(allocs)>>;


} // end ubu

#include "../../../detail/epilogue.hpp"

