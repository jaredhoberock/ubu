#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../detail/exception/throw_runtime_error.hpp"
#include "../event.hpp"
#include "has_runtime.hpp"
#include "kernel_entry_point.hpp"
#include "throw_on_error.hpp"
#include <cuda_runtime.h>
#include <set>
#include <stack>
#include <vector>


namespace ubu::cuda::detail
{


// node construction functions

inline cudaGraphNode_t make_empty_node(cudaGraph_t graph)
{
  cudaGraphNode_t result{};

  if UBU_TARGET(has_runtime())
  {
    throw_on_error(
      cudaGraphAddEmptyNode(&result, graph, nullptr, 0),
      "cuda::detail::make_empty_node: after cudaGraphAddEmptyNode"
    );
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::make_empty_node requires the CUDA Runtime.");
  }

  return result;
}


template<std::same_as<cudaGraphNode_t>... Nodes>
cudaGraphNode_t make_empty_node(cudaGraph_t graph, cudaGraphNode_t dependency, Nodes... dependencies)
{
  cudaGraphNode_t result{};

  if UBU_TARGET(has_runtime())
  {
    cudaGraphNode_t dependency_handles[] = {dependency, dependencies...};

    throw_on_error(
      cudaGraphAddEmptyNode(&result, graph, dependency_handles, 1 + sizeof...(dependencies)),
      "cuda::detail::make_empty_node: After cudaGraphAddEmptyNode"
    );
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::make_empty_node requires the CUDA Runtime.");
  }

  return result;
}


template<std::invocable F>
  requires std::is_trivially_copyable_v<F>
inline cudaGraphNode_t make_kernel_node(cudaGraph_t graph, cudaGraphNode_t dependency, dim3 grid_dim, dim3 block_dim, std::size_t dynamic_shared_memory_size, int device, F f)
{
  cudaGraphNode_t result{};

#if defined(__CUDACC__)

  if UBU_TARGET(has_runtime())
  {
    void* kernel_params[] = {reinterpret_cast<void*>(&f)};

    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(&kernel_entry_point<decltype(f)>);
    params.gridDim = grid_dim;
    params.blockDim = block_dim;
    params.kernelParams = kernel_params;
    params.sharedMemBytes = dynamic_shared_memory_size;

    temporarily_with_current_device(device, [&]
    {
      throw_on_error(
        cudaGraphAddKernelNode(&result, graph, &dependency, 1, &params),
        "cuda::detail::graph_context::make_kernel_node: After cudaGraphAddKernelNode"
      );
    });
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::make_kernel_node requires the CUDA Runtime.");
  }
#else
  ubu::detail::throw_runtime_error("cuda::detail:make_kernel_node requires CUDA C++ language support.");
#endif

  return result;
}



inline std::pair<cudaGraphNode_t, void*> make_mem_alloc_node(cudaGraph_t graph, cudaGraphNode_t dependency, int device, std::size_t num_bytes)
{
  cudaGraphNode_t result{};
  cudaMemAllocNodeParams params{};

  if UBU_TARGET(has_runtime())
  {
    params.poolProps.allocType = cudaMemAllocationTypePinned;
    params.poolProps.location.type = cudaMemLocationTypeDevice;
    params.poolProps.location.id = device;
    params.bytesize = num_bytes;

    throw_on_error(
      cudaGraphAddMemAllocNode(&result, graph, &dependency, 1, &params),
      "cuda::detail::make_mem_alloc_node: After cudaGraphAddMemAllocNode"
    );
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::make_mem_alloc_node requires the CUDA Runtime.");
  }

  return {result, params.dptr};
}


inline cudaGraphNode_t make_mem_free_node(cudaGraph_t graph, cudaGraphNode_t dependency, void* ptr)
{
  cudaGraphNode_t result{};

  if UBU_TARGET(has_runtime())
  {
    throw_on_error(
      cudaGraphAddMemFreeNode(&result, graph, &dependency, 1, ptr),
      "cuda::detail::make_mem_free_node: After cudaGraphAddMemFreeNode"
    );
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::make_mem_free_node requires the CUDA Runtime.");
  }

  return result;
}


inline cudaGraphNode_t make_memset_node(cudaGraph_t graph, cudaGraphNode_t dependency, void* ptr, unsigned char value, std::size_t num_bytes)
{
  cudaGraphNode_t result{};

  if UBU_TARGET(has_runtime())
  {
    cudaMemsetParams params
    {
      .dst = ptr,
      .pitch = 0, // unused
      .value = value,
      .elementSize = 1,
      .width = num_bytes,
      .height = 1,
    };

    throw_on_error(
      cudaGraphAddMemsetNode(&result, graph, &dependency, 1, &params),
      "cuda::detail::make_memset_node: After cudaGraphAddMemsetNode"
    );
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::make_memset_node requires the CUDA Runtime.");
  }

  return result;
}



// graph algorithms

inline std::vector<cudaGraphNode_t> dependencies_of(cudaGraphNode_t n)
{
  if UBU_TARGET(has_runtime())
  {
    std::size_t num_dependencies = 0;

    throw_on_error(
      cudaGraphNodeGetDependencies(n, nullptr, &num_dependencies),
      "cuda::detail::dependencies_of: After cudaGraphNodeGetDependencies"
    );

    std::vector<cudaGraphNode_t> result(num_dependencies);
    throw_on_error(
      cudaGraphNodeGetDependencies(n, result.data(), &num_dependencies),
      "cuda::detail::dependencies_of: After cudaGraphNodeGetDependencies"
    );

    return result;
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::dependencies_of requires the CUDA Runtime.");
  }

  return {};
}


inline std::vector<cudaGraphNode_t> dependent_nodes_of(cudaGraphNode_t n)
{
  if UBU_TARGET(has_runtime())
  {
    std::size_t num_dependent_nodes = 0;
    throw_on_error(
      cudaGraphNodeGetDependentNodes(n, nullptr, &num_dependent_nodes),
      "cuda::detail::dependent_nodes_of: After cudaGraphNodeGetDependentNodes"
    );

    std::vector<cudaGraphNode_t> result(num_dependent_nodes);
    throw_on_error(
      cudaGraphNodeGetDependentNodes(n, result.data(), &num_dependent_nodes),
      "cuda::detail::dependent_nodes_of: After cudaGraphNodeGetDependentNodes"
    );

    return result;
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::dependent_nodes_of requires the CUDA Runtime.");
  }

  return {};
}


inline std::set<cudaGraphNode_t> nodes_of_connected_subgraph(cudaGraphNode_t n)
{
  // find all nodes of n's subgraph
  std::set<cudaGraphNode_t> result;

  std::stack<cudaGraphNode_t> unvisited_nodes;

  unvisited_nodes.push(n);
  while(not unvisited_nodes.empty())
  {
    cudaGraphNode_t n = unvisited_nodes.top();
    unvisited_nodes.pop();

    if(!result.contains(n))
    {
      result.insert(n);

      for(auto dependency : dependencies_of(n))
      {
        unvisited_nodes.push(dependency);
      }

      for(auto dependent : dependent_nodes_of(n))
      {
        unvisited_nodes.push(dependent);
      }
    }
  }

  return result;
}


// this function searches a graph for all nodes disconnected from n
// and disables them so they are not executed when g is launched
//
// the result is that all events causally connected to n execute when g is launched
inline void disable_disconnected_nodes(cudaGraphExec_t executable, cudaGraph_t graph, cudaGraphNode_t n)
{
  if UBU_TARGET(has_runtime())
  {
    // find all nodes of n's connected subgraph
    auto to_enable = nodes_of_connected_subgraph(n);

    // disable all nodes disconnected from n
    std::size_t num_nodes = 0;
    throw_on_error(
      cudaGraphGetNodes(graph, nullptr, &num_nodes),
      "cuda::detail::disable_disconnected_nodes: After cudaGraphGetNodes"
    );

    std::vector<cudaGraphNode_t> all_nodes(num_nodes);
    throw_on_error(
      cudaGraphGetNodes(graph, all_nodes.data(), &num_nodes),
      "cuda::detail::disable_disconnected_nodes: After cudaGraphGetNodes"
    );

    // disable all nodes
    for(cudaGraphNode_t n : all_nodes)
    {
      // only disable kernel nodes for now
      cudaGraphNodeType type{};
      throw_on_error(
        cudaGraphNodeGetType(n, &type),
        "cuda::detail::disable_disconnected_nodes: After cudaGraphNodeGetType"
      );

      if(type == cudaGraphNodeTypeKernel)
      {
        throw_on_error(
          cudaGraphNodeSetEnabled(executable, n, false),
          "cuda::detail::disable_disconnected_nodes: After cudaGraphNodeSetEnabled"
        );
      }
    }

    // enable connected nodes
    for(cudaGraphNode_t n : to_enable)
    {
      // only enable kernel nodes for now
      cudaGraphNodeType type{};
      throw_on_error(
        cudaGraphNodeGetType(n, &type),
        "cuda::detail::disable_disconnected_nodes: After cudaGraphNodeGetType"
      );

      if(type == cudaGraphNodeTypeKernel)
      {
        throw_on_error(
          cudaGraphNodeSetEnabled(executable, n, true),
          "cuda::detail::disable_disconnected_nodes: After cudaGraphNodeSetEnabled"
        );
      }
    }
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::disable_disconnected_nodes requires the CUDA Runtime.");
  }
}


// graph instantiation

inline cudaGraphExec_t instantiate_and_enable_connected_subgraph(cudaGraph_t g, cudaGraphNode_t n)
{
  cudaGraphExec_t result{};

  if UBU_TARGET(has_runtime())
  {
    // we instantiate the graph with auto free on launch
    // because some allocations in the graph may not be matched with a deallocation
    // in the future, when support for disabling memory allocation and memory free nodes may exist,
    // we may wish to eliminate use of this flag

    throw_on_error(
      cudaGraphInstantiateWithFlags(&result, g, cudaGraphInstantiateFlagAutoFreeOnLaunch),
      "cuda::detail::instantiate_and_enable_connected_subgraph: After cudaGraphInstantiate"
    );

    // disable all nodes in the executable which cannot be reached from n

    disable_disconnected_nodes(result, g, n);
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::instantiate_and_enable_connected_subgraph requires the CUDA Runtime.");
  }

  return result;
}


// graph execution

inline cuda::event instantiate_and_enable_connected_subgraph_and_launch(cudaStream_t stream, cudaGraph_t graph, cudaGraphNode_t launched_from)
{
  cudaGraphExec_t executable_graph = instantiate_and_enable_connected_subgraph(graph, launched_from);

  if UBU_TARGET(has_runtime())
  {
    throw_on_error(
      cudaGraphLaunch(executable_graph, stream),
      "cuda::detail::instantiate_and_enable_connected_subgraph_and_launch: After cudaGraphLaunch"
    );

    throw_on_error(
      cudaGraphExecDestroy(executable_graph),
      "cuda::detail::instantiate_and_enable_connected_subgraph_and_launch: After cudaGraphExecDestroy"
    );
  }
  else
  {
    ubu::detail::throw_runtime_error("cuda::detail::instantiate_and_enable_connected_subgraph_and_launch requires the CUDA Runtime.");
  }

  // record the event on device 0
  return {0, stream};
}


} // end ubu::cuda::detail


#include "../../../detail/epilogue.hpp"

