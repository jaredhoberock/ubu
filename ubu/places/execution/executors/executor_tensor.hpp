#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../tensors/coordinates/concepts/bounded_coordinate.hpp"
#include "../../../tensors/coordinates/concepts/congruent.hpp"
#include "../../../tensors/coordinates/coordinate_cat.hpp"
#include "../../../tensors/coordinates/split_coordinate_at.hpp"
#include "../../../tensors/coordinates/traits/rank.hpp"
#include "../../../tensors/inplace_tensor.hpp"
#include "../../../tensors/iterators.hpp"
#include "../../../tensors/shapes/shape.hpp"
#include "../../../tensors/views/all.hpp"
#include "../../../tensors/views/enumerate_transform.hpp"
#include "../../../tensors/views/tile.hpp"
#include "../../../utilities/tuples.hpp"
#include "../../causality/after_all.hpp"
#include "../../causality/happening.hpp"
#include "../../memory/allocators/allocate_and_zero_after.hpp"
#include "../../memory/allocators/concepts/asynchronous_allocator.hpp"
#include "../../memory/views/reinterpret_buffer.hpp"
#include "bulk_execute_after.hpp"
#include "bulk_execute_with_workspace_after.hpp"
#include "concepts/executor.hpp"
#include "traits/executor_happening.hpp"
#include "traits/executor_shape.hpp"
#include "traits/executor_workspace.hpp"

#include <array>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace ubu
{

// XXX this thing does not actually satisfy tensor_like yet
template<executor E, bounded_coordinate S>
  requires std::is_trivially_copy_constructible_v<E>
class executor_tensor
{
  private:
    static constexpr std::size_t inner_rank = rank_v<executor_shape_t<E>>;

  public:
    using happening_type = executor_happening_t<E>;
    using shape_type = coordinate_cat_result_t<executor_shape_t<E>, S>;

    template<tensor_like OtherExecs>
      requires congruent<shape_t<OtherExecs>,S>
    constexpr executor_tensor(from_tensor_like_t, OtherExecs&& execs)
      : execs_(from_tensor_like, all(std::forward<OtherExecs>(execs)))
    {}

    template<std::size_t N>
      requires (rank_v<S> == 1)
    constexpr executor_tensor(std::array<E,N> execs)
      : executor_tensor(from_tensor_like, execs)
    {}

    template<std::same_as<E>... U>
      requires (rank_v<S> == 1)
    constexpr executor_tensor(E exec, U... execs)
      : executor_tensor(std::array<E,1+sizeof...(execs)>{{exec,execs...}})
    {}

    executor_tensor(const executor_tensor&) = default;

    executor_tensor() = default;

    template<congruent<shape_type> K, std::invocable<default_coordinate_t<K>> F>
      requires std::is_trivially_copy_constructible_v<F>
    constexpr happening auto bulk_execute_after(const happening_type& before, K kernel_shape, F f) const
    {
      auto [inner_shape, outer_shape] = split_coordinate_at<inner_rank>(kernel_shape);

      // we'll map bulk_execute_after over the tensor of executors
      auto xfrm = enumerate_transform(tile(execs_, outer_shape), [&](auto outer_coord, auto exec)
      {
        return ubu::bulk_execute_after(exec, before, inner_shape, [=](auto inner_coord)
        {
          auto coord = coordinate_cat(inner_coord, outer_coord);

          f(coord);
        });
      });

      // materialize the calls to bulk_execute_after into a vector of happenings
      std::vector happenings(begin(xfrm), end_iterator(xfrm));

      // "reduce" the happenings into a single happening
      // XXX it would be better if we could return the happenings tensor directly
      return after_all(std::move(happenings));
    }

    template<asynchronous_allocator A>
    struct workspace_type
    {
      using buffer_type = tuples::second_t<allocate_after_result_t<std::byte,A,happening_type,std::size_t>>;

      buffer_type buffer;
      executor_workspace_t<E,A> local_workspace;
    };

    template<asynchronous_allocator A>
    using workspace_shape_type = workspace_shape_t<workspace_type<A>>;

    template<asynchronous_allocator A, congruent<shape_type> K, congruent<workspace_shape_type<A>> W, std::invocable<default_coordinate_t<K>,workspace_type<A>> F>
      requires std::is_trivially_copy_constructible_v<F>
    constexpr happening auto bulk_execute_with_workspace_after(const A& alloc, const happening_type& before, K kernel_shape, W workspace_shape, F f) const
    {
      // decompose shapes
      auto [inner_shape, outer_shape] = split_coordinate_at<inner_rank>(kernel_shape);
      auto [inner_workspace_shape, outer_buffer_size] = split_coordinate_at<rank_v<W>-1>(workspace_shape);

      // allocate a zeroed outer buffer after the before event
      auto [outer_buffer_ready, outer_buffer] = allocate_and_zero_after<std::byte>(alloc, *this, before, outer_buffer_size);

      // we'll map bulk_execute_with_workspace_after over the tensor of executors
      view auto xfrm = enumerate_transform(tile(execs_, outer_shape), [&](auto outer_coord, auto exec)
      {
        return ubu::bulk_execute_with_workspace_after(exec, alloc, outer_buffer_ready, inner_shape, inner_workspace_shape, [=](auto inner_coord, auto inner_ws)
        {
          auto coord = coordinate_cat(inner_coord, outer_coord);

          workspace_type<A> ws{outer_buffer, inner_ws};

          f(coord, ws);
        });
      });

      // materialize the calls to bulk_execute_after_with_workspace into a vector of happenings
      std::vector happenings(begin(xfrm), end_iterator(xfrm));

      // deallocate the outer buffer after the kernel
      return deallocate_after(alloc, after_all(std::move(happenings)), outer_buffer);
    }

    constexpr happening auto execute_after(happening auto&& before, std::invocable auto f)
    {
      return execs_[0].execute_after(before, f);
    }

    constexpr bool operator==(const executor_tensor& other) const
    {
      return std::ranges::equal(execs_, other.execs_);
    }

    constexpr bool operator!=(const executor_tensor& other) const
    {
      return not (*this == other);
    }

    constexpr S shape() const
    {
      return ubu::shape(execs_);
    }

  private:
    inplace_tensor<E,S> execs_;
};

template<tensor_like Execs>
  requires shaped_and_bounded<Execs>
executor_tensor(from_tensor_like_t, Execs&&) -> executor_tensor<tensor_element_t<Execs&&>, tensor_shape_t<Execs&&>>;

template<executor E, std::same_as<E>... Execs>
executor_tensor(E exec, Execs... execs) -> executor_tensor<E, constant<1+sizeof...(execs)>>;

} // end ubu


#include "../../../detail/epilogue.hpp"

