#pragma once

#include "../../detail/prologue.hpp"
#include "../algorithms.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/traits/zeros.hpp"
#include "../element_exists.hpp"
#include "../iterators.hpp"
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


template<matrix_like M, congruent<shape_t<M>> S>
std::vector<int> compute_column_widths(const M& matrix, S max_shape)
{
  if(ubu::empty(matrix)) return {};

  auto [num_rows, num_cols] = shape(matrix);
  auto [max_rows, max_cols] = max_shape;

  bool do_column_elision = max_cols != 0 and max_cols < num_cols;
  bool do_row_elision    = max_rows != 0 and max_rows < num_rows;

  // when eliding elements, the output gets an additional row or column representing elision
  int num_output_rows = do_row_elision    ? max_rows + 1 : num_rows;
  int num_output_cols = do_column_elision ? max_cols + 1 : num_cols;

  // -2 because elision occurs on the second to last element
  int elided_row_idx = num_output_rows - 2;
  int elided_col_idx = num_output_cols - 2;

  std::vector<int> max_widths(num_output_cols);

  // iterate over elements of the output matrix
  for(int output_row_idx = 0; output_row_idx < num_output_rows; ++output_row_idx)
  {
    for(int output_col_idx = 0; output_col_idx < num_output_cols; ++output_col_idx)
    {
      std::size_t length = 0;

      // map our output column index to a column in the input matrix
      int input_col_idx = output_col_idx;
      if(do_column_elision and output_col_idx == elided_col_idx + 1)
      {
        input_col_idx = num_cols - 1;
      }

      if(do_row_elision and output_row_idx == elided_row_idx)
      {
        // XXX because vertical ellipsis is a 3-byte character, this will return 3 
        //length = std::formatted_size("{}", "⋮");
        length = 1;
      }
      else if(do_column_elision and output_col_idx == elided_col_idx)
      {
        length = std::formatted_size("{}", "...");
      }
      else
      {
        // map our output coordinate to a coordinate in the input matrix
        int input_row_idx = output_row_idx;
        if(do_row_elision and input_row_idx == elided_row_idx + 1)
        {
          input_row_idx = num_rows - 1;
        }

        int2 input_coord(input_row_idx, input_col_idx);

        if(ubu::element_exists(matrix, input_coord))
        {
          length = std::formatted_size("{}", matrix[input_coord]);
        }
        else
        {
          length = std::formatted_size("X");
        }
      }

      if(length > max_widths[output_col_idx])
      {
        max_widths[output_col_idx] = length;
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

  std::vector<int> column_widths = detail::compute_column_widths(matrix, max_shape);
  
  auto [num_rows, num_cols] = shape(matrix);
  auto [max_rows, max_cols] = max_shape;

  bool do_column_elision = max_cols != 0 and max_cols < num_cols;
  bool do_row_elision    = max_rows != 0 and max_rows < num_rows;

  // when eliding elements, the output gets an additional row or column representing elision
  int num_output_rows = do_row_elision    ? max_rows + 1 : num_rows;
  int num_output_cols = do_column_elision ? max_cols + 1 : num_cols;

  // -2 because elision occurs on the second to last element
  int elided_row_idx = num_output_rows - 2;
  int elided_col_idx = num_output_cols - 2;

  // Function to format a single element of the output
  auto format_output_element_to = [&](auto out, int output_row_idx, int output_col_idx)
  {
    int column_width = column_widths[output_col_idx];

    if(do_row_elision and output_row_idx == elided_row_idx)
    {
      if(do_column_elision and output_col_idx == elided_col_idx)
      {
        // on the row and column representing elided row & column in the input,
        // the single element is represented by a diagonal ellipsis

        // XXX for some reason, std::format_to thinks the diagnoal ellipsis consumes
        //     zero width, but padding out the column_width by 2 corrects the problem?
        out = std::format_to(out, " {:^{}} |", "⋱", column_width + 2);
      }
      else
      {
        // on the row representing elided rows in the input,
        // each element is represented by a vertical ellipsis

        // XXX for some reason, std::format_to thinks the vertical ellipsis consumes
        //     zero width, but padding out the column_width by 2 corrects the problem?
        out = std::format_to(out, " {:^{}} |", "⋮", column_width + 2);
      }
    }
    else if(do_column_elision and output_col_idx == elided_col_idx)
    {
      // in a column representing elided columns in the input,
      // each element is represented by a horizontal ellipsis
      out = std::format_to(out, " {:>{}} |", "...", column_width);
    }
    else
    {
      // map our output coordinate to a coordinate in the input matrix
      int input_row_idx = output_row_idx;
      if(do_row_elision and input_row_idx == elided_row_idx + 1)
      {
        input_row_idx = num_rows - 1;
      }

      // map our output column index to a column in the input matrix
      int input_col_idx = output_col_idx;
      if(do_column_elision and output_col_idx == elided_col_idx + 1)
      {
        input_col_idx = num_cols - 1;
      }

      int2 input_coord(input_row_idx, input_col_idx);

      if(ubu::element_exists(matrix, input_coord))
      {
        out = std::format_to(out, " {:>{}} |", matrix[input_coord], column_width);
      }
      else
      {
        out = std::format_to(out, " {:>{}} |", "X", column_width);
      }
    }

    return out;
  };
  
  // Function to format a border separating rows
  auto format_border_to = [&](auto out)
  {
    // the beginning of the border
    out = std::format_to(out, "+");
  
    for(size_t output_col_idx = 0; output_col_idx < num_output_cols; ++output_col_idx)
    {
      out = std::format_to(out, "{:-<{}}+", "", column_widths[output_col_idx] + 2);
    }
  
    // the border includes a newline
    return std::format_to(out, "\n");
  };

  // format top border
  out = format_border_to(out);

  // format each row of the output
  for(size_t row_idx = 0; row_idx < num_output_rows; ++row_idx)
  {
    // format left border
    out = std::format_to(out, "|");
    
    // format each column of the output
    for(size_t col_idx = 0; col_idx < num_output_cols; ++col_idx)
    {
      out = format_output_element_to(out, row_idx, col_idx);
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

