#pragma once

#include "../../detail/prologue.hpp"

#include "detail/graph_utility_functions.hpp"
#include <concepts>
#include <cuda_runtime_api.h>
#include <utility>


namespace ubu::cuda
{


class graph_node
{
  public:
    inline graph_node(cudaGraph_t graph, cudaGraphNode_t native_handle, cudaStream_t stream)
      : graph_{graph},
        native_handle_{native_handle},
        stream_{stream}
    {}

    inline graph_node(graph_node&& other) noexcept
      : graph_{},
        native_handle_{},
        stream_{}
    {
      std::swap(graph_, other.graph_);
      std::swap(native_handle_, other.native_handle_);
      std::swap(stream_, other.stream_);
    }

    inline cudaGraph_t graph() const
    {
      return graph_;
    }

    inline void wait()
    {
      detail::instantiate_and_enable_connected_subgraph_and_launch(stream_, graph(), native_handle()).wait();
    }

    template<std::same_as<graph_node>... Nodes>
    graph_node because_of(const Nodes&... nodes) const
    {
      return {graph(), detail::make_empty_node(graph_, native_handle(), nodes.native_handle()...), stream_};
    }

    inline cudaGraphNode_t native_handle() const
    {
      return native_handle_;
    }

  private:
    cudaGraph_t graph_;
    cudaGraphNode_t native_handle_;
    cudaStream_t stream_;
};


} // end ubu::cuda


#include "../../detail/epilogue.hpp"

