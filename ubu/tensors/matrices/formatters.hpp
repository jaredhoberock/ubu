#pragma once

#include "../../detail/prologue.hpp"
#include "../algorithms.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/traits/zeros.hpp"
#include "../element_exists.hpp"
#include "../shapes/shape.hpp"
#include "matrix_like.hpp"
#include "rows.hpp"
#include <format> // XXX we should probably guard use of this #include
#include <iterator>
#include <vector>

namespace ubu
{
namespace detail
{

// XXX to make matrix formatting work on the GPU, we would need to use "compiled" format strings in the following


// Function to get the maximum width of elements in each column of the matrix
template<matrix_like M>
std::vector<int> compute_column_widths(M&& matrix)
{
  if(ubu::empty(matrix)) return {};

  auto [num_rows, num_cols] = shape(matrix);
  
  std::vector<int> max_widths(num_cols, 0);

  for(auto row : rows(matrix))
  {
    for(auto col : domain(row))
    {
      std::size_t length = element_exists(row, col) ?
        std::formatted_size("{}", row[col]) :
        std::formatted_size("X")
      ;

      if(length > max_widths[col])
      {
        max_widths[col] = length;
      }
    }
  }

  return max_widths;
}

} // end detail


template<class OutputIt, matrix_like M, congruent<shape_t<M>> S = shape_t<M>>
constexpr OutputIt format_matrix_to(OutputIt out, const M& matrix, S max_shape = zeros_v<S>)
{
  if(ubu::empty(matrix)) return out;

  std::vector<int> column_widths = detail::compute_column_widths(matrix);
  
  auto [num_rows, num_cols] = shape(matrix);
  auto [max_rows, max_cols] = max_shape;

  bool do_column_elision = max_cols == 0 or max_cols < num_cols;
  bool do_row_elision    = max_rows == 0 or max_rows < num_rows;
  
  // Function to format a border separating rows
  auto format_border_to = [&](auto out)
  {
    // the beginning of the border
    out = std::format_to(out, "+");
  
    for(size_t col = 0; col < num_cols; ++col)
    {
      if(do_column_elision and col == max_cols - 2)
      {
        // 5 is the length of " ... ", which represents elided columns
        out = std::format_to(out, "{:-<{}}+", "", 5);

        // skip to the final column
        col = num_cols - 2;
      }
      else
      {
        out = std::format_to(out, "{:-<{}}+", "", column_widths[col] + 2);
      }
    }
  
    // the border includes a newline
    return std::format_to(out, "\n");
  };

  // format top border, measure its width
  auto format_border_result = format_border_to(std::counted_iterator(out, 0));

  // get the new value of output iterator
  out = format_border_result.base();

  // measure the border width, subtracting one for the newline
  // std::counted_iterator decrements its count each time it's written (wtf)
  // so to get the number of items written, we have to subtract from zero
  std::size_t border_width = (0 - format_border_result.count()) - 1;

  // format each row of the matrix
  for(size_t row_idx = 0; row_idx < num_rows; ++row_idx)
  {
    if(do_row_elision and row_idx == max_rows - 1)
    {
      // format the representation of elided rows: | ... |
      // the ellipsis is centered between two pipes
      out = std::format_to(out, "|{:^{}}|\n", "...", border_width - 2);

      // skip to the last row
      row_idx = num_rows - 1;

      // top border for the last row
      out = format_border_to(out);
    }
    
    // format left border
    out = std::format_to(out, "|");

    auto row = rows(matrix)[row_idx];
    
    // format each element of the row
    for(size_t col = 0; col < num_cols; ++col)
    {
      if(do_column_elision and col == max_cols - 2)
      {
        // represent elided columns
        out = std::format_to(out, " ... |");

        // skip to the final column
        col = num_cols - 2;
      }
      else
      {
        if(ubu::element_exists(row, col))
        {
          out = std::format_to(out, " {:>{}} |", row[col], column_widths[col]);
        }
        else
        {
          out = std::format_to(out, " {:>{}} |", "X", column_widths[col]);
        }
      }
    }

    out = std::format_to(out, "\n");
    
    // format border before next row
    out = format_border_to(out);
  }
  
  return out;
}


template<matrix_like M, congruent<shape_t<M>> S = shape_t<M>>
constexpr std::string format_matrix(const M& matrix, S max_shape = zeros_v<S>)
{
  std::string result;
  format_matrix_to(std::back_inserter(result), matrix, max_shape);
  return result;
}


namespace detail
{


template<ubu::matrix_like M>
struct matrix_formatter
{
  std::pair<size_t,size_t> max_shape{}; // defaults to output the entire matrix

  template<class ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    // XXX we would like to be able to parse max_shape out of the format string, somehow
    //     ChatGPT's suggestions didn't work
    return ctx.begin();
  }


  template<class FormatContext>
  constexpr auto format(const M& m, FormatContext& ctx) const
  {
    return ubu::format_matrix_to(ctx.out(), m, max_shape);
  }
};

} // end detail
} // end ubu


template<ubu::matrix_like M>
struct std::formatter<M> : ubu::detail::matrix_formatter<M> {};

#if __has_include(<fmt/format.h>)

#include <fmt/format.h>
#include <fmt/ranges.h>

template<ubu::matrix_like M>
struct fmt::formatter<M> : ubu::detail::matrix_formatter<M> {};

// disable fmtlib detecting matrix_like as a range
template<ubu::matrix_like M>
struct fmt::is_range<M,char> : std::false_type {};

#endif // __has_include(<fmt/format.h>)

#include "../../detail/epilogue.hpp"

