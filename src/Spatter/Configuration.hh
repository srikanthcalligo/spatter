/*!
  \file Configuration.hh
*/

#ifndef SPATTER_CONFIGURATION_HH
#define SPATTER_CONFIGURATION_HH

#include <algorithm>
#include <cctype>
#include <experimental/iterator>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "CudaBackend.hh"
#include <cuda.h>
#include <cuda_runtime_api.h>

// stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCudaErrors(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "checkCudaErrors: " << cudaGetErrorString(code) << " " << file
              << " " << line << std::endl;
    if (abort)
      exit(code);
  }
}
#endif

#include "AlignedAllocator.hh"
#include "SpatterTypes.hh"
#include "Timer.hh"

#define ALIGN 64
template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, ALIGN>>;


#ifdef USE_TT_METAL

#include <tt-metalium/host_api.hpp>
#include "tt-metalium/constants.hpp"
#include "tt-metalium/util.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/work_split.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/tile.hpp"
#include "tt-metalium/command_queue.hpp"
#include "tt-metalium/tilize_utils.hpp"

#include "MetaliumBackend.hh"
#endif


namespace Spatter {

class ConfigurationBase {
public:
  ConfigurationBase(const size_t id, const std::string name, std::string k,
      const aligned_vector<size_t> &pattern,
      const aligned_vector<size_t> &pattern_gather,
      const aligned_vector<size_t> &pattern_scatter,
      aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter,
      const long int seed, const size_t wrap, const size_t count,
      const size_t shared_mem, const size_t local_work_size, const int nthreads,
      const unsigned long nruns, const bool aggregate, const bool atomic,
      const unsigned long verbosity, size_t tt_compute_mode = 0);

  virtual ~ConfigurationBase();

  virtual int run(bool timed, unsigned long run_id);

  virtual void gather(bool timed, unsigned long run_id) = 0;
  virtual void scatter(bool timed, unsigned long run_id) = 0;
  virtual void scatter_gather(bool timed, unsigned long run_id) = 0;
  virtual void multi_gather(bool timed, unsigned long run_id) = 0;
  virtual void multi_scatter(bool timed, unsigned long run_id) = 0;

  virtual void report();

  virtual void setup();

private:
  void print_no_mpi(
      size_t bytes_per_run, double minimum_time, double maximum_bandwidth);

#ifdef USE_MPI
  void print_mpi(std::vector<unsigned long long> &vector_bytes_per_run,
      std::vector<double> &vector_minimum_time,
      std::vector<double> &vector_maximum_bandwidth);
#endif

public:
  const size_t id;
  const std::string name;

  std::string kernel;
  const aligned_vector<size_t> pattern;
  const aligned_vector<size_t> pattern_gather;
  const aligned_vector<size_t> pattern_scatter;

  aligned_vector<double> &sparse;
  double *&dev_sparse;
  size_t &sparse_size;

  aligned_vector<double> &sparse_gather;
  double *&dev_sparse_gather;
  size_t &sparse_gather_size;

  aligned_vector<double> &sparse_scatter;
  double *&dev_sparse_scatter;
  size_t &sparse_scatter_size;

  aligned_vector<double> &dense;
  aligned_vector<aligned_vector<double>> &dense_perthread;
  double *&dev_dense;
  size_t &dense_size;

  const size_t delta;
  const size_t delta_gather;
  const size_t delta_scatter;
  
  long int seed;
  const size_t wrap;
  const size_t count;

  size_t shmem;
  size_t local_work_size;

  const int omp_threads;
  const unsigned long nruns;

  const bool aggregate;
  const bool atomic;
  const unsigned long verbosity;
  //TT-Metalium : Status of Compute mode
  size_t tt_compute_mode;

  Spatter::Timer timer;
  std::vector<double> time_seconds;
};

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config);

template <typename Backend> class Configuration : public ConfigurationBase {};

template <> class Configuration<Spatter::Serial> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> &pattern,
      const aligned_vector<size_t> &pattern_gather,
      const aligned_vector<size_t> &pattern_scatter,
      aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter,
      const long int seed, const size_t wrap, const size_t count,
      const unsigned long nruns, const bool aggregate,
      const unsigned long verbosity);

  void gather(bool timed, unsigned long run_id);
  void scatter(bool timed, unsigned long run_id);
  void scatter_gather(bool timed, unsigned long run_id);
  void multi_gather(bool timed, unsigned long run_id);
  void multi_scatter(bool timed, unsigned long run_id);
};

#ifdef USE_OPENMP
template <> class Configuration<Spatter::OpenMP> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> &pattern,
      const aligned_vector<size_t> &pattern_gather,
      aligned_vector<size_t> &pattern_scatter,
      aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter,
      const long int seed, const size_t wrap, const size_t count,
      const int nthreads, const unsigned long nruns, const bool aggregate,
      const bool atomic, const unsigned long verbosity);

  int run(bool timed, unsigned long run_id);

  void gather(bool timed, unsigned long run_id);
  void scatter(bool timed, unsigned long run_id);
  void scatter_gather(bool timed, unsigned long run_id);
  void multi_gather(bool timed, unsigned long run_id);
  void multi_scatter(bool timed, unsigned long run_id);
};
#endif

#ifdef USE_CUDA
template <> class Configuration<Spatter::CUDA> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> &pattern,
      const aligned_vector<size_t> &pattern_gather,
      const aligned_vector<size_t> &pattern_scatter,
      aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter,
      const long int seed, const size_t wrap, const size_t count,
      const size_t shared_mem, const size_t local_work_size,
      const unsigned long nruns, const bool aggregate, const bool atomic,
      const unsigned long verbosity);

  ~Configuration();

  int run(bool timed, unsigned long run_id);
  void gather(bool timed, unsigned long run_id);
  void scatter(bool timed, unsigned long run_id);
  void scatter_gather(bool timed, unsigned long run_id);
  void multi_gather(bool timed, unsigned long run_id);
  void multi_scatter(bool timed, unsigned long run_id);
  void setup();

public:
  size_t *dev_pattern;
  size_t *dev_pattern_gather;
  size_t *dev_pattern_scatter;
};
#endif

#ifdef USE_TT_METAL
template <> class Configuration<Spatter::TT_Metalium> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> &pattern,
      const aligned_vector<size_t> &pattern_gather,
      const aligned_vector<size_t> &pattern_scatter,
      aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter,
      const long int seed, const size_t wrap, const size_t count,
      const unsigned long nruns, const bool aggregate,
      const unsigned long verbosity, size_t tt_compute_mode = 0, size_t tt_parallel_mode = 0, size_t tt_core_id = 0, size_t tt_step_size = 1, size_t tt_nr_enabled = 0);

~Configuration();

  int run(bool timed, unsigned long run_id);
  void gather(bool timed, unsigned long run_id);
  void scatter(bool timed, unsigned long run_id);
  void scatter_gather(bool timed, unsigned long run_id);
  void multi_gather(bool timed, unsigned long run_id);
  void multi_scatter(bool timed, unsigned long run_id);
  void setup();

private:
  double kernel_exec_time = 0.0;
  CoreCoord core = {0,0};
  CoreRangeSet core_set;
  uint32_t device_id = 0; 
  IDevice *device = CreateDevice(device_id);
  CommandQueue& cq = device->command_queue();
  Program program = CreateProgram();
  uint32_t single_tile_size = 32 * 32; //TILE_WIDTH * TILE_HEIGHT
  uint32_t num_tiles_per_cb = 1;
  size_t step_size;
  size_t is_nr_enabled;
  CBHandle cb_sparse;
  CBHandle cb_dense;
  CBHandle cb_pattern;
  CBHandle cb_intermediate;
  CBHandle cb_pattern_gather;
  CBHandle cb_pattern_scatter;
  CBHandle cb_sparse_gather;
  CBHandle cb_sparse_scatter;
  KernelHandle data_read_kernel_handle;
  KernelHandle data_write_kernel_handle;
  KernelHandle compute_kernel_handle;

  size_t is_compute_mode_on; //TT-Metalium : To run on the compute core
  uint32_t is_parallel_mode_on; //TT-Metalium : To run in parallel mode
  uint32_t core_id = 0; //TT-metalium : To select the specific core id.
  uint32_t is_first_run = 0;

};
#endif

} // namespace Spatter

#endif
