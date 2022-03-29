#pragma once

#include "../detail/prologue.hpp"

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <utility>


ASPERA_NAMESPACE_OPEN_BRACE


namespace detail
{


template<typename T>
  class uninitialized
{
  private:
    typename std::aligned_storage<
      sizeof(T),
      std::alignment_of<T>::value
    >::type storage;

    const T* ptr() const
    {
      const void *result = storage.data;
      return reinterpret_cast<const T*>(result);
    }

    T* ptr()
    {
      return reinterpret_cast<T*>(&storage);
    }

  public:
    // copy assignment
    uninitialized<T> &operator=(const T &other)
    {
      T& self = *this;
      self = other;
      return *this;
    }

    T& get()
    {
      return *ptr();
    }

    const T& get() const
    {
      return *ptr();
    }

    operator T& ()
    {
      return get();
    }

    operator const T&() const
    {
      return get();
    }

    template<typename... Args>
    void construct(Args&&... args)
    {
      ::new(ptr()) T(std::forward<Args>(args)...);
    }

    void destroy()
    {
      T& self = *this;
      self.~T();
    }
};


extern __shared__ int s_data_segment_begin[];


class os
{
  public:
    inline os(size_t max_data_segment_size)
      : program_break_(s_data_segment_begin),
        max_data_segment_size_(max_data_segment_size)
    {}


    inline int brk(void *end_data_segment)
    {
      if(end_data_segment <= program_break_)
      {
        program_break_ = end_data_segment;
        return 0;
      }

      return -1;
    }


    inline void *sbrk(size_t increment)
    {
      if(data_segment_size() + increment <= max_data_segment_size_)
      {
        program_break_ = reinterpret_cast<char*>(program_break_) + increment;
      } // end if
      else
      {
        return reinterpret_cast<void*>(-1);
      } // end else

      return program_break_;
    }


  private:
    inline size_t data_segment_size()
    {
      return reinterpret_cast<char*>(program_break_) - reinterpret_cast<char*>(s_data_segment_begin);
    } // end data_segment_size()


    void *program_break_;
    std::uint32_t max_data_segment_size_;
};


// only one instance of this class can logically exist per CTA, and its use is thread-unsafe
class singleton_unsafe_on_chip_allocator
{
  public:
    inline singleton_unsafe_on_chip_allocator(size_t max_data_segment_size)
      : os_(max_data_segment_size),
        heap_begin_(reinterpret_cast<block*>(os_.sbrk(0))),
        heap_end_(heap_begin_)
    {}
  
    inline void *allocate(size_t size)
    {
      size_t aligned_size = align8(size);
    
      block *prev = find_first_free_insertion_point(heap_begin_, heap_end_, aligned_size);
    
      block *b;
    
      if(prev != heap_end_ && (b = prev->next()) != heap_end_)
      {
        // can we split?
        if((b->size() - aligned_size) >= sizeof(block))
        {
          split_block(b, aligned_size);
        } // end if
    
        b->set_is_free(false);
      } // end if
      else
      {
        // nothing fits, extend the heap
        b = extend_heap(prev, aligned_size);
        if(b == heap_end_)
        {
          return 0;
        } // end if
      } // end else
    
      return b->data();
    } // end allocate()
  
  
    inline void deallocate(void *ptr)
    {
      if(ptr != 0)
      {
        block *b = get_block(ptr);
    
        // free the block
        b->set_is_free(true);
    
        // try to fuse the freed block the previous block
        if(b->prev() && b->prev()->is_free())
        {
          b = b->prev();
          fuse_block(b);
        } // end if
    
        // now try to fuse with the next block
        if(b->next() != heap_end_)
        {
          fuse_block(b);
        } // end if
        else
        {
          heap_end_ = b;
    
          // the the OS know where the new break is
          os_.brk(b);
        } // end else
      } // end if
    } // end deallocate()


  private:
    class block
    {
      public:
        inline size_t size() const
        {
          return size_;
        }

        void set_size(size_t sz)
        {
          size_ = sz;
        }

        inline block *prev() const
        {
          return prev_;
        }

        void set_prev(block *p)
        {
          prev_ = p;
        }

        inline block *next() const
        {
          return reinterpret_cast<block*>(reinterpret_cast<char*>(data()) + size());
        }

        inline bool is_free() const
        {
          return is_free_;
        }

        inline void set_is_free(bool f)
        {
          is_free_ = f;
        }

        inline void *data() const
        {
          return reinterpret_cast<char*>(const_cast<block*>(this)) + sizeof(block);
        }

      private:
        [[maybe_unused]] std::aligned_storage<2 * sizeof(size_t)>::type data_;

        // this packing ensures that sizeof(block) is compatible with 64b alignment, because:
        // on a 32b platform, sizeof(block) == 64b
        // on a 64b platform, sizeof(block) == 128b
        // XXX consider reducing sizeof(block) by avoiding the use of size_t and a pointer
        bool   is_free_ : 1;
        size_t size_    : 8 * sizeof(size_t) - 1;
        block *prev_;
    };
  
  
    os     os_;
    block *heap_begin_;
    block *heap_end_;
  
  
    inline void split_block(block *b, size_t size)
    {
      block *new_block;
    
      // emplace a new block within the old one's data segment
      new_block = reinterpret_cast<block*>(reinterpret_cast<char*>(b->data()) + size);
    
      // the new block's size is the old block's size less the size of the split less the size of a block
      new_block->set_size(b->size() - size - sizeof(block));
    
      new_block->set_prev(b);
      new_block->set_is_free(true);
    
      // the old block's size is the size of the split
      b->set_size(size);
    
      // link the old block to the new one
      if(new_block->next() != heap_end_)
      {
        new_block->next()->set_prev(new_block);
      }
    }
  
  
    inline bool fuse_block(block *b)
    {
      if(b->next() != heap_end_ && b->next()->is_free())
      {
        // increment b's size by sizeof(block) plus the next's block's data size
        b->set_size(sizeof(block) + b->next()->size() + b->size());
    
        if(b->next() != heap_end_)
        {
          b->next()->set_prev(b);
        }
    
        return true;
      }
    
      return false;
    }
  
  
    inline static block *get_block(void *data)
    {
      // the block metadata lives sizeof(block) bytes to the left of data
      return reinterpret_cast<block *>(reinterpret_cast<char *>(data) - sizeof(block));
    }
  
  
    inline static block *find_first_free_insertion_point(block *first, block *last, size_t size)
    {
      block *prev = last;
    
      while(first != last && !(first->is_free() && first->size() >= size))
      {
        prev = first;
        first = first->next();
      }
    
      return prev;
    }
  
  
    inline block *extend_heap(block *prev, size_t size)
    {
      // the new block goes at the current end of the heap
      block *new_block = heap_end_;
    
      // move the break to the right to accomodate both a block and the requested allocation
      if(os_.sbrk(sizeof(block) + size) == reinterpret_cast<void*>(-1))
      {
        // allocation failed
        return new_block;
      }
    
      // record the new end of the heap
      heap_end_ = reinterpret_cast<block*>(reinterpret_cast<char*>(heap_end_) + sizeof(block) + size);
    
      new_block->set_size(size);
      new_block->set_prev(prev);
      new_block->set_is_free(false);
    
      return new_block;
    }


    inline static size_t align8(size_t size)
    {
      return ((((size - 1) >> 3) << 3) + 8);
    }
};


class singleton_on_chip_allocator
{
  public:
    inline singleton_on_chip_allocator(size_t max_data_segment_size)
      : mutex_(),
        alloc_(max_data_segment_size)
    {}


    inline void *allocate(size_t size)
    {
      void *result;

      mutex_.lock();
      {
        result = alloc_.allocate(size);
      } // end critical section
      mutex_.unlock();

      return result;
    }


    inline void deallocate(void *ptr)
    {
      mutex_.lock();
      {
        alloc_.deallocate(ptr);
      } // end critical section
      mutex_.unlock();
    }


  private:
    class mutex
    {
      public:
        inline mutex()
          : in_use_(0)
        {}


        inline bool try_lock()
        {
          unsigned int expected = 1;
          return std::atomic_ref<unsigned int>(in_use_).compare_exchange_weak(expected, 0) != 0;
        }


        inline void lock()
        {
          // spin while waiting
          while(try_lock());
        }


        inline void unlock()
        {
          in_use_ = 0;
        }


      private:
        unsigned int in_use_;
    };


    mutex mutex_;
    singleton_unsafe_on_chip_allocator alloc_;
};


// XXX this costs 40 bytes of smem
//     we could reduce this by avoiding the use of 64b pointers in the data structure
__shared__  uninitialized<singleton_on_chip_allocator> s_on_chip_allocator;


} // end detail


namespace cuda
{


__device__ inline void init_on_chip_malloc(size_t max_data_segment_size)
{
  detail::s_on_chip_allocator.construct(max_data_segment_size);
}


__device__ inline void *on_chip_malloc(size_t size)
{
  return detail::s_on_chip_allocator.get().allocate(size);
}


__device__ inline void on_chip_free(void *ptr)
{
  return detail::s_on_chip_allocator.get().deallocate(ptr);
}


template<typename T>
__device__ inline T *on_chip_cast(T *ptr)
{
  extern __shared__ char s_begin[];
  return reinterpret_cast<T*>((reinterpret_cast<char*>(ptr) - s_begin) + s_begin);
}


__device__ inline void *shmalloc(size_t num_bytes)
{
  // first try on_chip_malloc
  void *result = on_chip_malloc(num_bytes);
  
  if(!result)
  {
    result = std::malloc(num_bytes);
  } // end if

  return result;
}


} // end cuda


namespace detail
{


// XXX WAR buggy __isGlobal
__device__ bool is_on_chip(const void *ptr)
{
  unsigned int ret = 0;

#if defined(__CUDACC__)
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.shared p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#endif
#endif // __CUDACC__

  return ret;
}


} // end detail


namespace cuda
{


__device__ inline void shfree(void *ptr)
{
  if(detail::is_on_chip(ptr))
  {
    on_chip_free(ptr);
  }
  else
  {
    std::free(ptr);
  }
}


} // end cuda


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

