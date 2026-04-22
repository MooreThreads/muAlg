/******************************************************************************
 * Targeted device_spmv repro cases for MUSA.
 *
 * This binary is intentionally standalone: it does not depend on the CUB test
 * harness being re-enabled in CMake. The goal is to reproduce two distinct
 * failure classes with deterministic inputs:
 *   1. single_tile_double        -> one merge tile, no segment_fixup kernel
 *   2. segment_fixup_float       -> multiple merge tiles, segment_fixup kernel runs
 *   3. zero_nnz_double           -> zero-filled matrix should short-circuit to zeros
 *   4. single_tile_full_double   -> exact single-tile double case must flush tail carry
 ******************************************************************************/

#include <cub/device/device_spmv.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_scan.cuh>

#include <musa_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

template <typename ValueT>
struct HostCSRMatrix
{
  std::vector<ValueT> values;
  std::vector<int> row_offsets;
  std::vector<int> column_indices;
  int num_rows;
  int num_cols;
  int num_nonzeros;

  HostCSRMatrix(int rows, int cols)
      : row_offsets(rows + 1, 0), num_rows(rows), num_cols(cols), num_nonzeros(0)
  {}

  void append_value(int row, int col, ValueT value)
  {
    ++num_nonzeros;
    ++row_offsets[row];
    column_indices.push_back(col);
    values.push_back(value);
  }

  void finalize()
  {
    std::vector<int> counts = row_offsets;
    row_offsets[0]          = 0;
    for (int i = 0; i < num_rows; ++i)
    {
      row_offsets[i + 1] = row_offsets[i] + counts[i];
    }
  }

  int get_row_offset(int row) const { return row_offsets[row]; }

  int get_row_num_nonzero(int row) const
  {
    return row_offsets[row + 1] - row_offsets[row];
  }
};

template <typename ValueT>
HostCSRMatrix<ValueT> make_random_csr_matrix(int num_rows,
                                             int num_cols,
                                             float target_fill_ratio)
{
  HostCSRMatrix<ValueT> mat(num_rows, num_cols);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int row = 0; row < num_rows; ++row)
  {
    for (int col = 0; col < num_cols; ++col)
    {
      if (dist(gen) < target_fill_ratio)
      {
        ValueT value;
        if constexpr (std::is_floating_point<ValueT>::value)
        {
          std::uniform_real_distribution<ValueT> val_dist(
              static_cast<ValueT>(-100), static_cast<ValueT>(100));
          value = val_dist(gen);
        }
        else
        {
          std::uniform_int_distribution<int64_t> val_dist(
              std::numeric_limits<ValueT>::min(),
              std::numeric_limits<ValueT>::max());
          value = static_cast<ValueT>(val_dist(gen));
        }
        mat.append_value(row, col, value);
      }
    }
  }

  mat.finalize();
  return mat;
}

template <typename ValueT>
std::vector<ValueT> make_random_vector(int len)
{
  std::vector<ValueT> vec(len);
  std::mt19937 gen(42);

  for (int i = 0; i < len; ++i)
  {
    if constexpr (std::is_floating_point<ValueT>::value)
    {
      std::uniform_real_distribution<ValueT> dist(
          static_cast<ValueT>(-100), static_cast<ValueT>(100));
      vec[i] = dist(gen);
    }
    else
    {
      std::uniform_int_distribution<int64_t> dist(
          std::numeric_limits<ValueT>::min(),
          std::numeric_limits<ValueT>::max());
      vec[i] = static_cast<ValueT>(dist(gen));
    }
  }
  return vec;
}

template <typename ValueT>
void compute_reference(const HostCSRMatrix<ValueT> &a,
                       const std::vector<ValueT> &x,
                       std::vector<ValueT> &y)
{
  for (int row = 0; row < a.num_rows; ++row)
  {
    const int row_offset = a.get_row_offset(row);
    const int row_length = a.get_row_num_nonzero(row);

    ValueT accum{};
    for (int i = 0; i < row_length; ++i)
    {
      const int col = a.column_indices[row_offset + i];
      const ValueT val = a.values[row_offset + i];
      accum += val * x[col];
    }
    y[row] = accum;
  }
}

template <typename ValueT>
bool almost_equal(
    ValueT lhs,
    ValueT rhs,
    typename std::enable_if<std::is_floating_point<ValueT>::value>::type * =
        nullptr)
{
  constexpr double rel_tol = 1e-3;
  constexpr double abs_tol = 1e-2;
  const double limit =
      rel_tol * (std::fabs(lhs) + std::fabs(rhs)) + abs_tol;
  return std::fabs(lhs - rhs) <= limit;
}

template <typename ValueT>
bool almost_equal(
    ValueT lhs,
    ValueT rhs,
    typename std::enable_if<!std::is_floating_point<ValueT>::value>::type * =
        nullptr)
{
  return lhs == rhs;
}

template <typename ValueT>
double to_double(ValueT value)
{
  return static_cast<double>(value);
}

template <typename ValueT>
bool compare_results(const std::vector<ValueT> &reference,
                     const std::vector<ValueT> &result)
{
  if (reference.size() != result.size())
  {
    std::cerr << "Size mismatch: ref=" << reference.size()
              << " result=" << result.size() << '\n';
    return false;
  }

  for (std::size_t i = 0; i < reference.size(); ++i)
  {
    if (!almost_equal(reference[i], result[i]))
    {
      std::cerr << "Mismatch at position " << i << ": ref="
                << to_double(reference[i]) << " got=" << to_double(result[i])
                << '\n';
      return false;
    }
  }

  return true;
}

bool check_musa(musaError_t error, const char *what)
{
  if (error == musaSuccess)
  {
    return true;
  }

  std::cerr << what << " failed: " << static_cast<int>(error) << '\n';
  return false;
}

enum class ReproCase
{
  single_tile_double,
  segment_fixup_float,
  zero_nnz_double,
  single_tile_full_double,
  warp_scan_kv_double,
  all,
};

struct Options
{
  int device = 0;
  ReproCase repro_case = ReproCase::all;
};

void print_usage(const char *argv0)
{
  std::cout << argv0
            << " [--device=N] [--case=single_tile_double|segment_fixup_float|"
               "zero_nnz_double|single_tile_full_double|warp_scan_kv_double|all]\n";
}

Options parse_args(int argc, char **argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h")
    {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg.rfind("--device=", 0) == 0)
    {
      options.device = std::stoi(arg.substr(std::strlen("--device=")));
      continue;
    }
    if (arg.rfind("--case=", 0) == 0)
    {
      const std::string value = arg.substr(std::strlen("--case="));
      if (value == "single_tile_double")
      {
        options.repro_case = ReproCase::single_tile_double;
      }
      else if (value == "segment_fixup_float")
      {
        options.repro_case = ReproCase::segment_fixup_float;
      }
      else if (value == "zero_nnz_double")
      {
        options.repro_case = ReproCase::zero_nnz_double;
      }
      else if (value == "single_tile_full_double")
      {
        options.repro_case = ReproCase::single_tile_full_double;
      }
      else if (value == "warp_scan_kv_double")
      {
        options.repro_case = ReproCase::warp_scan_kv_double;
      }
      else if (value == "all")
      {
        options.repro_case = ReproCase::all;
      }
      else
      {
        std::cerr << "Unknown case: " << value << '\n';
        print_usage(argv[0]);
        std::exit(2);
      }
      continue;
    }

    std::cerr << "Unknown argument: " << arg << '\n';
    print_usage(argv[0]);
    std::exit(2);
  }

  return options;
}

struct WarpScanKvSnapshot
{
  int input_key;
  double input_value;
  int exclusive_key;
  double exclusive_value;
  int aggregate_key;
  double aggregate_value;
  int shuffle_key;
  double shuffle_value;
};

__global__ void warp_scan_kv_double_kernel(WarpScanKvSnapshot *out)
{
  using Pair = cub::KeyValuePair<int, double>;
  using WarpScanT = cub::WarpScan<Pair, 32>;

  __shared__ typename WarpScanT::TempStorage temp_storage;

  const int lane = threadIdx.x;
  Pair input;
  if (lane == 0)
  {
    input = Pair(15, -16839.814225);
  }
  else if (lane == 1)
  {
    input = Pair(16, 0.0);
  }
  else
  {
    input = Pair(16 + (lane * 4), 0.0);
  }

  Pair exclusive;
  Pair aggregate;
  WarpScanT(temp_storage).ExclusiveScan(
      input, exclusive, cub::ReduceByKeyOp<cub::Sum>(), aggregate);

  const Pair shuffled = cub::ShuffleUp<32>(input, 1, 0, 0xffffffffu);

  out[lane].input_key = input.key;
  out[lane].input_value = input.value;
  out[lane].exclusive_key = exclusive.key;
  out[lane].exclusive_value = exclusive.value;
  out[lane].aggregate_key = aggregate.key;
  out[lane].aggregate_value = aggregate.value;
  out[lane].shuffle_key = shuffled.key;
  out[lane].shuffle_value = shuffled.value;
}

bool run_warp_scan_kv_double_case()
{
  std::cout << "\n[case] warp_scan_kv_double\n";

  WarpScanKvSnapshot *d_snapshots = nullptr;
  std::vector<WarpScanKvSnapshot> h_snapshots(32);
  bool success = false;

  do
  {
    if (!check_musa(musaMalloc(&d_snapshots,
                               h_snapshots.size() * sizeof(WarpScanKvSnapshot)),
                    "musaMalloc(d_snapshots)"))
      break;

    if (!check_musa(musaMemset(d_snapshots,
                               0,
                               h_snapshots.size() * sizeof(WarpScanKvSnapshot)),
                    "musaMemset(d_snapshots)"))
      break;

    warp_scan_kv_double_kernel<<<1, 32>>>(d_snapshots);
    if (!check_musa(musaPeekAtLastError(), "warp_scan_kv_double_kernel launch"))
      break;
    if (!check_musa(musaDeviceSynchronize(),
                    "warp_scan_kv_double_kernel sync"))
      break;

    if (!check_musa(musaMemcpy(h_snapshots.data(),
                               d_snapshots,
                               h_snapshots.size() * sizeof(WarpScanKvSnapshot),
                               musaMemcpyDeviceToHost),
                    "musaMemcpy(warp_scan snapshots)"))
      break;

    for (int lane = 0; lane < 4; ++lane)
    {
      const auto &snapshot = h_snapshots[lane];
      std::cout << "lane=" << lane
                << " input={" << snapshot.input_key << ", "
                << snapshot.input_value << "}"
                << " shuffle={" << snapshot.shuffle_key << ", "
                << snapshot.shuffle_value << "}"
                << " exclusive={" << snapshot.exclusive_key << ", "
                << snapshot.exclusive_value << "}"
                << " aggregate={" << snapshot.aggregate_key << ", "
                << snapshot.aggregate_value << "}\n";
    }

    const auto &lane1 = h_snapshots[1];
    success = (lane1.shuffle_key == 15) &&
              almost_equal(lane1.shuffle_value, -16839.814225) &&
              (lane1.exclusive_key == 15) &&
              almost_equal(lane1.exclusive_value, -16839.814225);
  } while (false);

  musaFree(d_snapshots);

  std::cout << (success ? "[result] PASS" : "[result] FAIL") << '\n';
  return success;
}

template <typename ValueT>
bool run_case(const char *label, int rows, int cols, float fill_ratio)
{
  HostCSRMatrix<ValueT> h_a =
      make_random_csr_matrix<ValueT>(rows, cols, fill_ratio);
  std::vector<ValueT> h_x = make_random_vector<ValueT>(cols);
  std::vector<ValueT> h_y_reference(rows, ValueT{});
  compute_reference(h_a, h_x, h_y_reference);

  std::cout << "\n[case] " << label << '\n';
  std::cout << "rows=" << rows << " cols=" << cols
            << " nnz=" << h_a.num_nonzeros
            << " fill_ratio=" << fill_ratio << '\n';

  ValueT *d_values = nullptr;
  int *d_row_offsets = nullptr;
  int *d_column_indices = nullptr;
  ValueT *d_x = nullptr;
  ValueT *d_y = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  bool success = false;

  do
  {
    if (!check_musa(
            musaMalloc(&d_values, h_a.num_nonzeros * sizeof(ValueT)),
            "musaMalloc(d_values)"))
      break;
    if (!check_musa(
            musaMalloc(&d_row_offsets, (rows + 1) * sizeof(int)),
            "musaMalloc(d_row_offsets)"))
      break;
    if (!check_musa(
            musaMalloc(&d_column_indices, h_a.num_nonzeros * sizeof(int)),
            "musaMalloc(d_column_indices)"))
      break;
    if (!check_musa(musaMalloc(&d_x, cols * sizeof(ValueT)),
                    "musaMalloc(d_x)"))
      break;
    if (!check_musa(musaMalloc(&d_y, rows * sizeof(ValueT)),
                    "musaMalloc(d_y)"))
      break;

    if (h_a.num_nonzeros > 0)
    {
      if (!check_musa(musaMemcpy(d_values,
                                 h_a.values.data(),
                                 h_a.num_nonzeros * sizeof(ValueT),
                                 musaMemcpyHostToDevice),
                      "musaMemcpy(values)"))
        break;
      if (!check_musa(musaMemcpy(d_column_indices,
                                 h_a.column_indices.data(),
                                 h_a.num_nonzeros * sizeof(int),
                                 musaMemcpyHostToDevice),
                      "musaMemcpy(column_indices)"))
        break;
    }

    if (!check_musa(musaMemcpy(d_row_offsets,
                               h_a.row_offsets.data(),
                               (rows + 1) * sizeof(int),
                               musaMemcpyHostToDevice),
                    "musaMemcpy(row_offsets)"))
      break;
	    if (cols > 0)
	    {
	      if (!check_musa(musaMemcpy(d_x,
	                                 h_x.data(),
	                                 cols * sizeof(ValueT),
                                 musaMemcpyHostToDevice),
	                      "musaMemcpy(x)"))
	        break;
	    }

	    if (rows > 0)
	    {
	      if (!check_musa(musaMemset(d_y, 0, rows * sizeof(ValueT)),
	                      "musaMemset(y)"))
	        break;
	    }
	
	    auto error = cub::DeviceSpmv::CsrMV(d_temp_storage,
	                                        temp_storage_bytes,
	                                        d_values,
                                        d_row_offsets,
                                        d_column_indices,
                                        d_x,
                                        d_y,
                                        rows,
                                        cols,
                                        h_a.num_nonzeros);
    if (!check_musa(error, "DeviceSpmv::CsrMV(size query)"))
      break;

    if (!check_musa(musaMalloc(&d_temp_storage, temp_storage_bytes),
                    "musaMalloc(d_temp_storage)"))
      break;

    error = cub::DeviceSpmv::CsrMV(d_temp_storage,
                                   temp_storage_bytes,
                                   d_values,
                                   d_row_offsets,
                                   d_column_indices,
                                   d_x,
                                   d_y,
                                   rows,
                                   cols,
                                   h_a.num_nonzeros,
                                   0,
                                   true);
    if (!check_musa(error, "DeviceSpmv::CsrMV(run)"))
      break;

    if (!check_musa(musaDeviceSynchronize(), "musaDeviceSynchronize"))
      break;

    std::vector<ValueT> h_y(rows, ValueT{});
    if (rows > 0)
    {
      if (!check_musa(musaMemcpy(h_y.data(),
                                 d_y,
                                 rows * sizeof(ValueT),
                                 musaMemcpyDeviceToHost),
                      "musaMemcpy(y)"))
        break;
    }

    success = compare_results(h_y_reference, h_y);
  } while (false);

  musaFree(d_temp_storage);
  musaFree(d_y);
  musaFree(d_x);
  musaFree(d_column_indices);
  musaFree(d_row_offsets);
  musaFree(d_values);

  std::cout << (success ? "[result] PASS" : "[result] FAIL") << '\n';
  return success;
}

int main(int argc, char **argv)
{
  const Options options = parse_args(argc, argv);
  if (!check_musa(musaSetDevice(options.device), "musaSetDevice"))
  {
    return 1;
  }

  std::cout << "device_spmv targeted repro\n";
  std::cout << "device=" << options.device << '\n';

  bool pass = true;

  if (options.repro_case == ReproCase::single_tile_double ||
      options.repro_case == ReproCase::all)
  {
    pass &= run_case<double>("single_tile_double", 16, 16, 0.5f);
  }

  if (options.repro_case == ReproCase::segment_fixup_float ||
      options.repro_case == ReproCase::all)
  {
    pass &= run_case<float>("segment_fixup_float", 100, 100, 0.1f);
  }

  if (options.repro_case == ReproCase::zero_nnz_double ||
      options.repro_case == ReproCase::all)
  {
    pass &= run_case<double>("zero_nnz_double", 98, 84, 0.0f);
  }

  if (options.repro_case == ReproCase::single_tile_full_double ||
      options.repro_case == ReproCase::all)
  {
    pass &= run_case<double>("single_tile_full_double", 128, 2, 1.0002f);
  }

  if (options.repro_case == ReproCase::warp_scan_kv_double ||
      options.repro_case == ReproCase::all)
  {
    pass &= run_warp_scan_kv_double_case();
  }

  return pass ? 0 : 1;
}
