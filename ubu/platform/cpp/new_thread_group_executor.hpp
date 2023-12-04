#pragma once

#include "../../detail/prologue.hpp"

#include "new_thread_executor.hpp"
#include <barrier>
#include <cstddef>
#include <functional>
#include <span>
#include <thread>


namespace ubu
{
inline namespace cpp
{

struct new_thread_group_executor: public new_thread_executor
{
  struct workspace_type
  {
    std::span<std::byte> buffer;
    std::barrier<>& barrier;
  };

  template<std::invocable<std::size_t, workspace_type> F>
  std::future<void> new_bulk_execute_after(std::future<void>&& before, std::size_t n, std::size_t workspace_size, F&& f) const
  {
    return this->execute_after(std::move(before), [=,f=std::forward<F>(f)]
    {
      // the leader thread creates a workspace
      std::vector<std::byte> buffer(workspace_size);
      std::barrier<> barrier(n);
      workspace_type workspace{std::span(buffer), barrier};

      // spawn the rest of the group's threads
      std::vector<std::thread> threads;
      for(std::size_t i = 1; i != n; ++i)
      {
        threads.emplace_back([=]
        {
          std::invoke(f, i, workspace);
        });
      }

      // invoke for the leader
      f(0, workspace);

      // join the group
      for(auto& th : threads)
      {
        th.join();
      }
    });
  }
};

} // end cpp
} // end ubu

#include "../../detail/epilogue.hpp"

