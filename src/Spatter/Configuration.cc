/*!
  \file Configuration.cc
*/

#include <numeric>
#include <atomic>

#include "Configuration.hh"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/work_split.hpp"
//#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
//#include "tt_metal/impl/device/device.hpp"


using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

namespace Spatter {

ConfigurationBase::ConfigurationBase(const size_t id, const std::string name,
    std::string k, const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread, double *&dev_dense,
    size_t &dense_size, const size_t delta, const size_t delta_gather,
    const size_t delta_scatter, const long int seed, const size_t wrap,
    const size_t count, const size_t shared_mem, const size_t local_work_size,
    const int nthreads, const unsigned long nruns, const bool aggregate,
    const bool atomic, const unsigned long verbosity)
    : id(id), name(name), kernel(k), pattern(pattern),
      pattern_gather(pattern_gather), pattern_scatter(pattern_scatter),
      sparse(sparse), dev_sparse(dev_sparse), sparse_size(sparse_size),
      sparse_gather(sparse_gather), dev_sparse_gather(dev_sparse_gather),
      sparse_gather_size(sparse_gather_size), sparse_scatter(sparse_scatter),
      dev_sparse_scatter(dev_sparse_scatter),
      sparse_scatter_size(sparse_scatter_size), dense(dense),
      dense_perthread(dense_perthread), dev_dense(dev_dense),
      dense_size(dense_size), delta(delta), delta_gather(delta_gather),
      delta_scatter(delta_scatter), seed(seed), wrap(wrap), count(count),
      shmem(shared_mem), local_work_size(local_work_size),
      omp_threads(nthreads), nruns(nruns), aggregate(aggregate), atomic(atomic),
      verbosity(verbosity), time_seconds(nruns, 0) {
  std::transform(kernel.begin(), kernel.end(), kernel.begin(),
      [](unsigned char c) { return std::tolower(c); });
}

ConfigurationBase::~ConfigurationBase() = default;

int ConfigurationBase::run(bool timed, unsigned long run_id) {
  if (kernel.compare("gather") == 0)
    gather(timed, run_id);
  else if (kernel.compare("scatter") == 0)
    scatter(timed, run_id);
  else if (kernel.compare("sg") == 0)
    scatter_gather(timed, run_id);
  else if (kernel.compare("multigather") == 0)
    multi_gather(timed, run_id);
  else if (kernel.compare("multiscatter") == 0)
    multi_scatter(timed, run_id);
  else {
    std::cerr << "Invalid Kernel Type" << std::endl;
    return -1;
  }

  return 0;
}

void ConfigurationBase::report() {
  size_t bytes_moved = 0;

  if (kernel.compare("gather") == 0 || kernel.compare("scatter") == 0)
    bytes_moved = pattern.size() * count * sizeof(size_t);

  if (kernel.compare("sg") == 0)
    bytes_moved = (pattern_scatter.size() + pattern_gather.size()) * count * sizeof(size_t);

  if (kernel.compare("multiscatter") == 0)
    bytes_moved = pattern_scatter.size() * count * sizeof(size_t);

  if (kernel.compare("multigather") == 0)
    bytes_moved = pattern_gather.size() * count * sizeof(size_t);

#ifdef USE_MPI
  int numpes = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numpes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<unsigned long long> vector_bytes_per_run(numpes, 0);
  MPI_Gather(&bytes_moved, 1, MPI_UNSIGNED_LONG_LONG,
      vector_bytes_per_run.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
      MPI_COMM_WORLD);

  assert(nruns == time_seconds.size());
  std::vector<double> total_time_seconds(nruns, 0.0);
  MPI_Allreduce(time_seconds.data(), total_time_seconds.data(),
      static_cast<int>(nruns), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  long int index = std::distance(total_time_seconds.begin(),
      std::min_element(total_time_seconds.begin(), total_time_seconds.end()));
  assert(index >= 0);
  size_t min_index = static_cast<size_t>(index);

  double mpi_minimum_time = time_seconds[min_index];
  std::vector<double> vector_minimum_time(numpes, 0.0);
  MPI_Gather(&mpi_minimum_time, 1, MPI_DOUBLE, vector_minimum_time.data(), 1,
      MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double mpi_maximum_bandwidth =
      static_cast<double>(bytes_per_run) / mpi_minimum_time / 1000000.0;
  std::vector<double> vector_maximum_bandwidth(numpes, 0.0);
  MPI_Gather(&mpi_maximum_bandwidth, 1, MPI_DOUBLE,
      vector_maximum_bandwidth.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0)
    print_mpi(
        vector_bytes_per_run, vector_minimum_time, vector_maximum_bandwidth);
#else
  double min_time = *std::min_element(time_seconds.begin(), time_seconds.end());
  double bandwidth = static_cast<double>(bytes_moved) / min_time / 1000000.0;

  print_no_mpi(bytes_moved, min_time, bandwidth);
#endif
}

void ConfigurationBase::setup() {
  if (kernel.compare("multigather") == 0) {
    if (pattern.size() == 0) {
      std::cerr << "Pattern needs to have length of at least 1" << std::endl;
      exit(1);
    }
    if (pattern_gather.size() == 0) {
      std::cerr << "Pattern-Gather needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
  } else if (kernel.compare("multiscatter") == 0) {
    if (pattern.size() == 0) {
      std::cerr << "Pattern needs to have length of at least 1" << std::endl;
      exit(1);
    }
    if (pattern_scatter.size() == 0) {
      std::cerr << "Pattern-Scatter needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
  } else if (kernel.compare("sg") == 0) {
    if (pattern_gather.size() == 0) {
      std::cerr << "Pattern-Gather needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
    if (pattern_scatter.size() == 0) {
      std::cerr << "Pattern-Scatter needs to have length of at least 1"
                << std::endl;
      exit(1);
    }
    if (pattern_scatter.size() != pattern_gather.size()) {
      std::cerr
          << "Pattern-Scatter needs to be the same length as Pattern-gather"
          << std::endl;
      exit(1);
    }
  } else {
    if (pattern.size() == 0) {
      std::cerr << "Pattern needs to have length of at least 1" << std::endl;
      exit(1);
    }
  }

  // Gather and Scatter
  // dense size = pattern.size() * wrap
  // sparse size = max_pattern_val + delta * (count - 1) + 1
  //
  // Concurrent
  // sparse_scatter size = max_pattern_scatter_val + delta_scatter * (count -
  // 1) + 1 sparse_gather size = max_pattern_gather_val + delta_gather *
  // (count - 1) + 1
  //
  // MultiGather
  // dense size = pattern.size() * wrap
  // sparse size = max_pattern_val + delta * (count - 1) + 1
  // assert(pattern.size() > max_pattern_gather_val + 1)
  //
  // MultiScatter
  // dense size = pattern.size() * wrap
  // sparse size = max_pattern_val + delta * (count - 1) + 1
  // assert(pattern.size() > max_pattern_scatter_val + 1)

  if (kernel.compare("sg") == 0) {
    size_t max_pattern_scatter_val = *(std::max_element(
        std::cbegin(pattern_scatter), std::cend(pattern_scatter)));
    size_t max_pattern_gather_val = *(std::max_element(
        std::cbegin(pattern_gather), std::cend(pattern_gather)));
    size_t sparse_scatter_size_ =
        max_pattern_scatter_val + delta_scatter * (count - 1) + 1;
    size_t sparse_gather_size_ =
        max_pattern_gather_val + delta_gather * (count - 1) + 1;

    if (sparse_scatter_size < sparse_scatter_size_)
      sparse_scatter_size = sparse_scatter_size_;

    if (sparse_gather_size < sparse_gather_size_)
      sparse_gather_size = sparse_gather_size_;

    if (verbosity >= 3)
      std::cout << "Pattern Gather Array Size: " << pattern_gather.size()
                << "Pattern Scatter Array Size: " << pattern_scatter.size()
                << "\tDelta: " << delta << "\tCount: " << count
                << "\tWrap: " << wrap
                << "\tSparse Scatter Array Size: " << sparse_scatter_size
                << "\tSparse Gather Array Size: " << sparse_gather_size
                << "\tMax Pattern Scatter Val: " << max_pattern_scatter_val
                << "\tMax Pattern Gather Val: " << max_pattern_gather_val
                << std::endl;
  } else {
    const size_t max_pattern_val =
        *(std::max_element(std::begin(pattern), std::end(pattern)));
    const size_t dense_size_ = pattern.size() * wrap;
    const size_t sparse_size_ = max_pattern_val + delta * (count - 1) + 1;

    if (dense_size < dense_size_)
      dense_size = dense_size_;

    if (sparse_size < sparse_size_)
      sparse_size = sparse_size_;

    if (kernel.compare("multiscatter") == 0) {
      const size_t max_pattern_scatter_val = *(std::max_element(
          std::begin(pattern_scatter), std::end(pattern_scatter)));
      if (pattern.size() <= max_pattern_scatter_val) {
        std::cerr << "Pattern only has length " << pattern.size()
                  << " but needs to have length of at least "
                     "max_pattern_scatter_val = "
                  << max_pattern_scatter_val << std::endl;
        exit(1);
      }
    }

    if (kernel.compare("multigather") == 0) {
      const size_t max_pattern_gather_val = *(std::max_element(
          std::begin(pattern_gather), std::end(pattern_gather)));
      if (pattern.size() <= max_pattern_gather_val) {
        std::cerr << "Pattern only has length " << pattern.size()
                  << " but needs to have length of at least "
                     "max_pattern_gather_val = "
                  << max_pattern_gather_val << std::endl;
        exit(1);
      }
    }

    if (verbosity >= 3) {
      std::cout << "Pattern Array Size: " << pattern.size()
                << "\tDelta: " << delta << "\tCount: " << count
                << "\tWrap: " << wrap << "\tDense Array Size: " << dense_size
                << "\tSparse Array Size: " << sparse_size
                << "\tMax Pattern Val: " << max_pattern_val;

      if (kernel.compare("multiscatter") == 0)
        std::cout << "\tMax Pattern Scatter Val: "
                  << *(std::max_element(std::begin(pattern_scatter),
                         std::end(pattern_scatter)));

      if (kernel.compare("multigather") == 0)
        std::cout << "\tMax Pattern Gather Val: "
                  << *(std::max_element(
                         std::begin(pattern_gather), std::end(pattern_gather)));

      std::cout << std::endl;
    }
  }
}

void ConfigurationBase::print_no_mpi(
    size_t bytes_per_run, double minimum_time, double maximum_bandwidth) {
  std::cout << std::setw(15) << std::left << id << std::setw(15) << std::left
            << bytes_per_run << std::setw(15) << std::left << minimum_time
            << std::setw(15) << std::left << maximum_bandwidth << std::endl;
}

#ifdef USE_MPI
void ConfigurationBase::print_mpi(
    std::vector<unsigned long long> &vector_bytes_per_run,
    std::vector<double> &vector_minimum_time,
    std::vector<double> &vector_maximum_bandwidth) {

  unsigned long long total_bytes = std::accumulate(vector_bytes_per_run.begin(),
      vector_bytes_per_run.end(),
      std::remove_reference_t<decltype(vector_bytes_per_run)>::value_type(0));
  double average_bytes_per_rank = static_cast<double>(total_bytes) /
      static_cast<double>(vector_bytes_per_run.size());

  double total_minimum_time = std::accumulate(vector_minimum_time.begin(),
      vector_minimum_time.end(),
      std::remove_reference_t<decltype(vector_minimum_time)>::value_type(0));
  double average_minimum_time_per_rank =
      total_minimum_time / static_cast<double>(vector_minimum_time.size());

  double total_maximum_bandwidth = std::accumulate(
      vector_maximum_bandwidth.begin(), vector_maximum_bandwidth.end(),
      std::remove_reference_t<decltype(vector_maximum_bandwidth)>::value_type(
          0));
  double average_maximum_bandwidth_per_rank = total_maximum_bandwidth /
      static_cast<double>(vector_maximum_bandwidth.size());

  std::cout << std::setw(15) << std::left << id << std::setw(30) << std::left
            << average_bytes_per_rank << std::setw(30) << std::left
            << total_bytes << std::setw(30) << std::left
            << average_minimum_time_per_rank << std::setw(30) << std::left
            << average_maximum_bandwidth_per_rank << std::setw(30) << std::left
            << total_maximum_bandwidth << std::endl;

  if (verbosity >= 3) {
    std::cout << "\nBytes per rank\n";
    for (unsigned long long bytes : vector_bytes_per_run)
      std::cout << bytes << ' ';
    std::cout << '\n';

    std::cout << "Minimum time per rank(s)\n";
    for (double t : vector_minimum_time)
      std::cout << t << ' ';
    std::cout << '\n';

    std::cout << "Maximum bandwidth per rank(MB/s)\n";
    for (double bw : vector_maximum_bandwidth)
      std::cout << bw << ' ';
    std::cout << std::endl;
  }
}
#endif

std::ostream &operator<<(std::ostream &out, const ConfigurationBase &config) {
  std::stringstream config_output;

  config_output << "{";

  config_output << "'id': " << config.id << ", ";

  if (config.name.compare("") != 0)
    config_output << "'name': '" << config.name << "', ";

  config_output << "'kernel': '" << config.kernel << "', ";

  config_output << "'pattern': [";
  std::copy(std::begin(config.pattern), std::end(config.pattern),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'pattern-gather': [";
  std::copy(std::begin(config.pattern_gather), std::end(config.pattern_gather),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'pattern-scatter': [";
  std::copy(std::begin(config.pattern_scatter),
      std::end(config.pattern_scatter),
      std::experimental::make_ostream_joiner(config_output, ", "));
  config_output << "], ";

  config_output << "'delta': " << config.delta << ", ";
  config_output << "'delta-gather': " << config.delta_gather << ", ";
  config_output << "'delta-scatter': " << config.delta_scatter << ", ";

  config_output << "'count': " << config.count << ", ";

  if (config.seed > 0)
    config_output << "'seed': " << config.seed << ", ";

  if (config.aggregate)
    config_output << "'agg (nruns)': " << config.nruns << ", ";

  config_output << "'wrap': " << config.wrap << ", ";

  config_output << "'threads': " << config.omp_threads;

  config_output << "}";
  return out << config_output.str();
}

Configuration<Spatter::Serial>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread,
    double *&dev_dense, size_t &dense_size,const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const unsigned long nruns,
    const bool aggregate, const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather,
          delta_scatter, seed, wrap, count, 0, 1024, 1, nruns, aggregate, false,
          verbosity) {
  ConfigurationBase::setup();
}

void Configuration<Spatter::Serial>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef TT_SPATTER_ENABLE
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    uint32_t n_tiles = (count * pattern_length)/ 1024; //1024 = 32 * 32
    uint32_t buf_size = sizeof(uint32_t) * 1024 * n_tiles;
    constexpr uint32_t dest_buf_size = sizeof(uint32_t) * 1024;
    constexpr uint32_t buf_page_size = sizeof(uint32_t) * 1024;

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer1 = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer2 = CreateBuffer(buffer_config_b);

    auto src0_coord =  dram_buffer->noc_coordinates();
    auto src1_coord = dram_buffer1->noc_coordinates();
    auto dst_coord = dram_buffer2->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = CB::c_in0;

    constexpr uint32_t num_tiles_per_cb = 2;
    constexpr uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        buf_src0,
        {{src0_cb_index, tt::DataFormat::UInt32}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        dest_buf_size,
        {{src1_cb_index, tt::DataFormat::UInt32}}).set_page_size(src1_cb_index, buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );


    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        dest_buf_size,
        {{dst_cb_index, tt::DataFormat::UInt32}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
#ifdef SPATTER_RISCV_KERNEL
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/gather_kernel_in_riscv.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

#else
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/read_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/write_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter/src/Spatter/kernels/compute/gather.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
#endif
    //Declare pattern and sparse arrays
    std::vector<uint32_t> sparse_vec(sparse.size());
    std::vector<uint32_t> pattern_mat_val(pattern_length);
#ifdef PRINT_DEBUG
    printf("\nInput : ");
#endif
    for(int i=0 ; i < (1024*n_tiles) ; i++){
      sparse_vec[i] = sparse[i];
      //Printing last 8 elements to validate the final data
#ifdef PRINT_DEBUG
      if(i > ((1024*n_tiles) - 9))
        printf("%u ", sparse_vec[i]);
#endif
    }

#ifdef PRINT_DEBUG
    printf("\n");
#endif
    for(int i=0 ; i < pattern_length ; i++){
      pattern_mat_val[i] = pattern[i];
    }
    EnqueueWriteBuffer(cq, dram_buffer, sparse_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer1, pattern_mat_val, false);

#ifdef SPATTER_RISCV_KERNEL
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles, dram_buffer2->address(), dst_coord.x, dst_coord.y, pattern_length, delta, wrap});
#else
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles});
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(),src0_coord.x, src0_coord.y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), dst_coord.x, dst_coord.y, n_tiles});
    //, dram_buffer1->address()
    //SetRuntimeArgs(program, void_data_kernel_noc1, core, {dram_buffer1->address()});
#endif
    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //printf("Hello, device 0, handle the data\n");

    vector<uint32_t> result;
    EnqueueReadBuffer(cq, dram_buffer2, result, true);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
#ifdef PRINT_DEBUG
    //printf("Destination array size = %zu\n", result.size());
    printf("Final : ");
    for(int i=0 ; i < pattern_length ; i++){
      printf("%u ", result[i]);
    }
    printf("\n\n");
#endif
    CloseDevice(device);
#else
  if (timed)
    timer.start();

  //printf("Count = %zu pattern_length = %zu %zu %zu\n", count, pattern_length, sparse.size(), dense.size());
  for (size_t i = 0; i < count; ++i){
    for (size_t j = 0; j < pattern_length; ++j){
      dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
    }
  }
#ifdef PRINT_DEBUG
  printf("RHost : ");
  for(int i=0 ; i < pattern_length ; i++){
    printf("%u ", (uint32_t)dense[i]);
  }
  printf("\n\n");
#endif
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
#endif
}

void Configuration<Spatter::Serial>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef TT_SPATTER_ENABLE
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    //Define and create buffers with uint32_t data type
    uint32_t n_tiles = (count * pattern_length)/ 1024; //1024 = 32 * 32
    uint32_t buf_size = sizeof(uint32_t) * 1024;
    uint32_t dest_buf_size = sizeof(uint32_t) * 1024 * n_tiles;
    constexpr uint32_t buf_page_size = sizeof(uint32_t) * 1024;

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer1 = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer2 = CreateBuffer(buffer_config_b);

    auto src0_coord =  dram_buffer->noc_coordinates();
    auto src1_coord = dram_buffer1->noc_coordinates();
    auto dst_coord = dram_buffer2->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = CB::c_in0;

    constexpr uint32_t num_tiles_per_cb = 2;
    constexpr uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        buf_size,
        {{src0_cb_index, tt::DataFormat::UInt32}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        buf_size,
        {{src1_cb_index, tt::DataFormat::UInt32}}).set_page_size(src1_cb_index, buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );


    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        buf_src0,
        {{dst_cb_index, tt::DataFormat::UInt32}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
#ifdef SPATTER_RISCV_KERNEL
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/scatter_kernel_in_riscv.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

#else
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/read_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/write_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter/src/Spatter/kernels/compute/gather.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
#endif
    //Input arrays
    std::vector<uint32_t> sparse_vec(pattern_length);
    std::vector<uint32_t> pattern_mat_val(pattern_length);
#ifdef PRINT_DEBUG
    printf("\nInput : ");
#endif
    for(int i=0 ; i < dense.size() ; i++){
      sparse_vec[i] = dense[i];
#ifdef PRINT_DEBUG
      printf("%u ", sparse_vec[i]);
#endif
    }
#ifdef PRINT_DEBUG
    printf("\n");
#endif
    for(int i=0 ; i < pattern_length ; i++){
      pattern_mat_val[i] = pattern[i];
    }
    EnqueueWriteBuffer(cq, dram_buffer, sparse_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer1, pattern_mat_val, false);

#ifdef SPATTER_RISCV_KERNEL
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles, dram_buffer2->address(), dst_coord.x, dst_coord.y, pattern_length, delta, wrap});
#else
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles});
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(),src0_coord.x, src0_coord.y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), dst_coord.x, dst_coord.y, n_tiles});
#endif
    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //printf("Hello, device 0, handle the data\n");

    std::vector<uint32_t> dense_vec(1024 * n_tiles);
    EnqueueReadBuffer(cq, dram_buffer2, dense_vec, true);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
#ifdef PRINT_DEBUG  
    //printf("Destination array size = %zu\n", result.size());
    printf("Final : ");
    for(int i=0 ; i < pattern_length ; i++){
      printf("%u ", dense_vec[i]);
    }
    printf("\n\n");
#endif
    CloseDevice(device);

//End of TT_SPATTER_ENABLE
#else

#ifdef TT_SPATTER_PARALLEL_ENABLE
    constexpr uint32_t device_id = 0; 
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    /*
    * Multi-Core prep
    */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    //auto compute_with_storage_grid_size = device->grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    printf("cc %u %u\n", num_cores_x, num_cores_y);

    constexpr CoreCoord core = {0,0};
    //Define and create buffers with uint32_t data type
    uint32_t n_tiles = (count * pattern_length)/ 1024; //1024 = 32 * 32
    uint32_t buf_size = sizeof(uint32_t) * 1024;
    uint32_t dest_buf_size = sizeof(uint32_t) * 1024 * n_tiles;
    constexpr uint32_t buf_page_size = sizeof(uint32_t) * 1024;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);

    std::cout << num_cores << std::endl;

    std::cout << core_group_1.num_cores() << std::endl;
    std::cout << core_group_2.num_cores() << std::endl;
    if(core_group_2.num_cores() > 0){
      printf("Core group2 Error.  TBD\n");
      exit(0);
    }
    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer1 = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer2 = CreateBuffer(buffer_config_b);

    auto src0_coord =  dram_buffer->noc_coordinates();
    auto src1_coord = dram_buffer1->noc_coordinates();
    auto dst_coord = dram_buffer2->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = CB::c_in0;

    constexpr uint32_t num_tiles_per_cb = 2;
    constexpr uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src0_cb_index, tt::DataFormat::UInt32}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src1_cb_index, tt::DataFormat::UInt32}}).set_page_size(src1_cb_index, buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb1_src1_config
    );


    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{dst_cb_index, tt::DataFormat::UInt32}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        cb_dst_config
    );

    //Create datamovement kernels
#ifdef SPATTER_RISCV_KERNEL_PARALLEL
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/scatter_kernel_in_riscv_multicore.cpp",
                    all_cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

#else
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/read_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/write_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter/src/Spatter/kernels/compute/gather.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
#endif
    //Input pattern and sparse arrary
    std::vector<uint32_t> sparse_vec(pattern_length);
    std::vector<uint32_t> pattern_mat_val(pattern_length);
    
    for(int i=0 ; i < dense.size() ; i++){
      sparse_vec[i] = dense[i];
    }
  
    for(int i=0 ; i < pattern_length ; i++){
      pattern_mat_val[i] = pattern[i];
    }
    EnqueueWriteBuffer(cq, dram_buffer, sparse_vec, false);
    EnqueueWriteBuffer(cq, dram_buffer1, pattern_mat_val, false);

#ifdef SPATTER_RISCV_KERNEL_PARALLEL
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        //if (core_group_1.contains(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        //} else if (core_group_2.contains(core)) {
        //    num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        //} else {
        //    TT_ASSERT(false, "Core not in specified core ranges");
        //}
        SetRuntimeArgs(program,
         void_data_kernel_noc0_read,
         core,
         {dram_buffer->address(),
         dram_buffer1->address(),
         src0_coord.x,
         src0_coord.y,
         src1_coord.x,
         src1_coord.y,
         n_tiles,
         dram_buffer2->address(),
         dst_coord.x,
         dst_coord.y,
         pattern_length,
         delta,
         wrap,
         num_tiles_written,
         num_output_tiles_per_core, 
         i});
        num_tiles_written += num_output_tiles_per_core;
    }
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles, dram_buffer2->address(), dst_coord.x, dst_coord.y, pattern_length, delta, wrap});
#else
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles});
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(),src0_coord.x, src0_coord.y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), dst_coord.x, dst_coord.y, n_tiles});
#endif
    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //printf("Hello, device 0, handle the data\n");

    std::vector<uint32_t> dense_vec(1024 * n_tiles);
    EnqueueReadBuffer(cq, dram_buffer2, dense_vec, true);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
#ifdef PRINT_DEBUG 
    printf("Destination array size = %zu\n", dense_vec.size());
    printf("Final : ");
    for(int i=dense_vec.size()-8 ; i < dense_vec.size() ; i++){
      printf("%u ", dense_vec[i]);
    }
    printf("\n\n");
#endif
    CloseDevice(device);
//Host Code
#else
  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j)
      sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];

#ifdef PRINT_DEBUG
  printf("HFinal : ");
  for(int i=0 ; i < pattern_length ; i++){
      printf("%u ", (uint32_t)sparse[i]);
  }
#endif

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
#endif //TT_SPATTER_PARALLEL_ENABLE
#endif //TT_SPATTER_ENABLE
}

void Configuration<Spatter::Serial>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef TT_SPATTER_ENABLE
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    uint32_t n_tiles = std::ceil(sparse_gather.size() / (double)1024); //(count * 8)/ 1024; //1024 = 32 * 32
    uint32_t buf_size = sizeof(uint32_t) * 1024;
    uint32_t dest_buf_size = sizeof(uint32_t) * 1024 * n_tiles;
    constexpr uint32_t buf_page_size = sizeof(uint32_t) * 1024;

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> pattern_scatter_dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> pattern_gather_dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> sparse_scatter_dram_buffer = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> sparse_gather_dram_buffer = CreateBuffer(buffer_config_b);

    auto pscatter_coord =  pattern_scatter_dram_buffer->noc_coordinates();
    auto pgather_coord = pattern_gather_dram_buffer->noc_coordinates();
    auto sscatter_coord = sparse_scatter_dram_buffer->noc_coordinates();
    auto sgather_coord = sparse_gather_dram_buffer->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = CB::c_in0;

    constexpr uint32_t num_tiles_per_cb = 2;
    constexpr uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src0_cb_index, tt::DataFormat::UInt32}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src1_cb_index, tt::DataFormat::UInt32}}).set_page_size(src1_cb_index, buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );

    constexpr uint32_t src2_cb_index = tt::CB::c_in2;

    CircularBufferConfig cb1_src2_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src2_cb_index, tt::DataFormat::UInt32}}).set_page_size(src2_cb_index, buf_page_size);

    CBHandle cb_id2 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src2_config
    );


    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{dst_cb_index, tt::DataFormat::UInt32}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
#ifdef SPATTER_RISCV_KERNEL
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/scatter_gather_kernel_in_riscv.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

#else
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/read_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/write_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter/src/Spatter/kernels/compute/gather.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
#endif
    //Input arrarys
    std::vector<uint32_t> pattern_scatter_dram(pattern_length);
    std::vector<uint32_t> pattern_gather_dram(pattern_length);
    std::vector<uint32_t> sparse_gather_dram(1024 * n_tiles);
    std::vector<uint32_t> sparse_scatter_dram(1024 * n_tiles);

    for(int i=0 ; i < sparse_gather.size(); i++){
      sparse_gather_dram[i] = sparse_gather[i];
    }
    for(int i=0 ; i < pattern_length ; i++){
      pattern_scatter_dram[i] = pattern_scatter[i];
      pattern_gather_dram[i] = pattern_gather[i];
    }
    EnqueueWriteBuffer(cq, pattern_gather_dram_buffer, pattern_gather_dram, false);
    EnqueueWriteBuffer(cq, pattern_scatter_dram_buffer, pattern_scatter_dram, false);
    EnqueueWriteBuffer(cq, sparse_gather_dram_buffer, sparse_gather_dram, false);

#ifdef SPATTER_RISCV_KERNEL
    SetRuntimeArgs(
      program,
      void_data_kernel_noc0_read,
      core, {pattern_gather_dram_buffer->address(),pgather_coord.x, pgather_coord.y,
             pattern_scatter_dram_buffer->address(),pscatter_coord.x, pscatter_coord.y,
             sparse_gather_dram_buffer->address(),sgather_coord.x, sgather_coord.y,
             sparse_scatter_dram_buffer->address(), sscatter_coord.x, sscatter_coord.y,
             n_tiles,pattern_length, delta_gather, delta_scatter, count});
#else
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles});
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(),src0_coord.x, src0_coord.y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), dst_coord.x, dst_coord.y, n_tiles});
#endif
    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //printf("Hello, device 0, handle the data\n");
    EnqueueReadBuffer(cq, sparse_scatter_dram_buffer, sparse_scatter_dram, true);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
#ifdef PRINT_DEBUG
    //printf("Destination array size = %zu\n", result.size());
    printf("Final : ");
    for(int i=(count-5) ; i < count ; i++){
      printf("%u ", sparse_scatter_dram[pattern_scatter_dram[0] + delta_scatter * i]);
    }
    printf("\n\n");
#endif
    CloseDevice(device);
#else
  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i){
    for (size_t j = 0; j < pattern_length; ++j){
      sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
          sparse_gather[pattern_gather[j] + delta_gather * i];
    }
  }
#ifdef PRINT_DEBUG
  printf("HFinal : ");
  for(int i=(count-5) ; i < count ; i++){
      printf("%f ", sparse_scatter[pattern_scatter[0] + delta_scatter * i]);
  }
  printf("\n");
#endif
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
#endif
}

void Configuration<Spatter::Serial>::multi_gather(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef TT_SPATTER_ENABLE
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    //Define and create buffers with uint32_t data type
    uint32_t n_tiles = std::ceil(sparse.size() / (double)1024); //(count * 8)/ 1024; //1024 = 32 * 32
    uint32_t sparse_buf_size = sizeof(uint32_t) * 1024 * n_tiles;
    uint32_t dest_buf_size = sizeof(uint32_t) * 1024;
    constexpr uint32_t buf_page_size = sizeof(uint32_t) * 1024;

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = sparse_buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> pattern_dram_buffer = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> pattern_gather_dram_buffer = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> sparse_dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dense_dram_buffer = CreateBuffer(buffer_config_b);

    auto pattern_coord =  pattern_dram_buffer->noc_coordinates();
    auto pgather_coord = pattern_gather_dram_buffer->noc_coordinates();
    auto sparse_coord = sparse_dram_buffer->noc_coordinates();
    auto dense_coord = dense_dram_buffer->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = CB::c_in0;

    constexpr uint32_t num_tiles_per_cb = 2;
    constexpr uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src0_cb_index, tt::DataFormat::UInt32}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src1_cb_index, tt::DataFormat::UInt32}}).set_page_size(src1_cb_index, buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );

    constexpr uint32_t src2_cb_index = tt::CB::c_in2;

    CircularBufferConfig cb1_src2_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src2_cb_index, tt::DataFormat::UInt32}}).set_page_size(src2_cb_index, buf_page_size);

    CBHandle cb_id2 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src2_config
    );


    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{dst_cb_index, tt::DataFormat::UInt32}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
#ifdef SPATTER_RISCV_KERNEL
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/multi_gather_kernel_in_riscv.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

#else
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/read_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/write_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter/src/Spatter/kernels/compute/gather.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
#endif
    //uint32_t pattern and sparse arrary
    std::vector<uint32_t> pattern_dram(pattern.size());
    std::vector<uint32_t> pattern_gather_dram(pattern_gather.size());
    std::vector<uint32_t> sparse_dram(sparse.size());
    std::vector<uint32_t> dense_dram(dense.size());

    for(int i=0 ; i < sparse.size(); i++){
      sparse_dram[i] = sparse[i];
    }
    for(int i=0 ; i < pattern.size() ; i++){
      pattern_dram[i] = pattern[i];
    }

    for(int i=0 ; i < pattern_gather.size() ; i++){
      pattern_gather_dram[i] = pattern_gather[i];
    }

    EnqueueWriteBuffer(cq, pattern_dram_buffer, pattern_dram, false);
    EnqueueWriteBuffer(cq, pattern_gather_dram_buffer, pattern_gather_dram, false);
    EnqueueWriteBuffer(cq, sparse_dram_buffer, sparse_dram, false);

#ifdef SPATTER_RISCV_KERNEL
    SetRuntimeArgs(
      program,
      void_data_kernel_noc0_read,
      core, {pattern_gather_dram_buffer->address(),pgather_coord.x, pgather_coord.y,
             pattern_dram_buffer->address(),pattern_coord.x, pattern_coord.y,
             sparse_dram_buffer->address(),sparse_coord.x, sparse_coord.y,
             dense_dram_buffer->address(), dense_coord.x, dense_coord.y,
             n_tiles,pattern_length, delta, wrap, pattern.size()});
#else
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles});
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(),src0_coord.x, src0_coord.y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), dst_coord.x, dst_coord.y, n_tiles});
#endif
    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //printf("Hello, device 0, handle the data\n");
    EnqueueReadBuffer(cq, dense_dram_buffer, dense_dram, true);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
#ifdef PRINT_DEBUG
    //printf("Destination array size = %zu\n", result.size());
    printf("Final : ");
    for(int i=0 ; i < 1 ; i++){
      printf("%u ", dense_dram[i]);
    }
    printf("\n\n");
#endif
    CloseDevice(device);
#else
  if (timed)
    timer.start();

  //printf("%zu %zu %zu %zu %zu %zu %zu %zu\n", dense.size(), sparse.size(), pattern.size(), pattern_gather.size(), count, pattern_length, delta, wrap);
  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j)
    {
      dense[j + pattern_length * (i % wrap)] =
          sparse[pattern[pattern_gather[j]] + delta * i];
    }
#ifdef PRINT_DEBUG
  printf("HFinal : ");
  for(int i=0 ; i < 1 ; i++){
      printf("%f ", dense[i]);
  }
  printf("\n\n");
#endif
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
#endif
}

void Configuration<Spatter::Serial>::multi_scatter(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef TT_SPATTER_ENABLE
    constexpr CoreCoord core = {0,0};
    constexpr uint32_t device_id = 0; 
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();

    //Define and create buffers with uint32_t data type
    uint32_t n_tiles = std::ceil(sparse.size() / (double)1024); //(count * 8)/ 1024; //1024 = 32 * 32
    uint32_t sparse_buf_size = sizeof(uint32_t) * 1024 * n_tiles;
    uint32_t dest_buf_size = sizeof(uint32_t) * 1024;
    constexpr uint32_t buf_page_size = sizeof(uint32_t) * 1024;

    //printf("No.of Tiles = %u\n", n_tiles);

    tt_metal::BufferConfig buffer_config_a = {
            .device = device,
            .size = sparse_buf_size ,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::BufferConfig buffer_config_b = {
            .device = device,
            .size = dest_buf_size,
            .page_size = buf_page_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> pattern_dram_buffer = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> pattern_scatter_dram_buffer = CreateBuffer(buffer_config_b);
    std::shared_ptr<tt::tt_metal::Buffer> sparse_dram_buffer = CreateBuffer(buffer_config_a);
    std::shared_ptr<tt::tt_metal::Buffer> dense_dram_buffer = CreateBuffer(buffer_config_b);

    auto pattern_coord =  pattern_dram_buffer->noc_coordinates();
    auto pscatter_coord = pattern_scatter_dram_buffer->noc_coordinates();
    auto sparse_coord = sparse_dram_buffer->noc_coordinates();
    auto dense_coord = dense_dram_buffer->noc_coordinates();

    //Create circular buffer to move data from DRAM to L1

    constexpr uint32_t src0_cb_index = CB::c_in0;

    constexpr uint32_t num_tiles_per_cb = 2;
    constexpr uint32_t buf_src0 = num_tiles_per_cb * buf_page_size;
    CircularBufferConfig cb0_src0_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src0_cb_index, tt::DataFormat::UInt32}}).set_page_size(src0_cb_index, buf_page_size);
    
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_src0_config);


    constexpr uint32_t src1_cb_index = tt::CB::c_in1;

    CircularBufferConfig cb1_src1_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src1_cb_index, tt::DataFormat::UInt32}}).set_page_size(src1_cb_index, buf_page_size);

    CBHandle cb_id1 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src1_config
    );

    constexpr uint32_t src2_cb_index = tt::CB::c_in2;

    CircularBufferConfig cb1_src2_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{src2_cb_index, tt::DataFormat::UInt32}}).set_page_size(src2_cb_index, buf_page_size);

    CBHandle cb_id2 = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb1_src2_config
    );


    constexpr uint32_t dst_cb_index = tt::CB::c_out0;

    CircularBufferConfig cb_dst_config = CircularBufferConfig(
        num_tiles_per_cb * buf_page_size,
        {{dst_cb_index, tt::DataFormat::UInt32}}).set_page_size(dst_cb_index, buf_page_size);

    CBHandle cb_out = tt_metal::CreateCircularBuffer(
        program,
        core,
        cb_dst_config
    );

    //Create datamovement kernels
#ifdef SPATTER_RISCV_KERNEL
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/multi_scatter_kernel_in_riscv.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

#else
    KernelHandle void_data_kernel_noc0_read = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/read_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    KernelHandle void_data_kernel_noc1_write = CreateKernel(
                    program,
                    "tt_metal/programming_examples/spatter/src/Spatter/kernels/data/write_kernel.cpp",
                    core,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/spatter/src/Spatter/kernels/compute/gather.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        }
    );
#endif
    //uint32_t pattern and sparse arrarys
    std::vector<uint32_t> pattern_dram(pattern.size());
    std::vector<uint32_t> pattern_scatter_dram(pattern_scatter.size());
    std::vector<uint32_t> sparse_dram(sparse.size());
    std::vector<uint32_t> dense_dram(dense.size());

    for(int i=0 ; i < dense.size(); i++){
      dense_dram[i] = dense[i];
    }
    for(int i=0 ; i < pattern.size() ; i++){
      pattern_dram[i] = pattern[i];
    }

    for(int i=0 ; i < pattern_gather.size() ; i++){
      pattern_scatter_dram[i] = pattern_scatter[i];
    }

    EnqueueWriteBuffer(cq, pattern_dram_buffer, pattern_dram, false);
    EnqueueWriteBuffer(cq, pattern_scatter_dram_buffer, pattern_scatter_dram, false);
    EnqueueWriteBuffer(cq, dense_dram_buffer, dense_dram, false);

#ifdef SPATTER_RISCV_KERNEL
    SetRuntimeArgs(
      program,
      void_data_kernel_noc0_read,
      core, {pattern_scatter_dram_buffer->address(),pscatter_coord.x, pscatter_coord.y,
             pattern_dram_buffer->address(),pattern_coord.x, pattern_coord.y,
             sparse_dram_buffer->address(),sparse_coord.x, sparse_coord.y,
             dense_dram_buffer->address(), dense_coord.x, dense_coord.y,
             n_tiles,pattern_length, delta, wrap, pattern.size()});
#else
    SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(), dram_buffer1->address(),src0_coord.x, src0_coord.y, src1_coord.x, src1_coord.y, n_tiles});
    //SetRuntimeArgs(program, void_data_kernel_noc0_read, core, {dram_buffer->address(),src0_coord.x, src0_coord.y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, void_data_kernel_noc1_write, core, {dram_buffer2->address(), dst_coord.x, dst_coord.y, n_tiles});
#endif
    if (timed)
      timer.start();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //printf("Hello, device 0, handle the data\n");
    EnqueueReadBuffer(cq, sparse_dram_buffer, sparse_dram, true);
    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
#ifdef PRINT_DEBUG
    //printf("Destination array size = %zu\n", result.size());
    printf("Final : ");
    for(int i=1 ; i < 10 ; i=i+pattern.size()){
      printf("%u ", sparse_dram[i]);
    }
    printf("\n\n");
#endif
    CloseDevice(device);
#else
  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j){
      sparse[pattern[pattern_scatter[j]] + delta * i] =
          dense[j + pattern_length * (i % wrap)];
    }
#ifdef PRINT_DEBUG
  printf("HFinal : ");
  for(int i=1 ; i < 10 ; i=i+pattern.size()){
      printf("%f ", sparse[i]);
  }
  printf("\n\n");
#endif
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

#endif
}

#ifdef USE_OPENMP
Configuration<Spatter::OpenMP>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread,
    double *&dev_dense, size_t &dense_size,const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const int nthreads,
    const unsigned long nruns, const bool aggregate, const bool atomic,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather, delta_scatter, seed, wrap,
          count, 0, 1024, nthreads, nruns, aggregate, atomic, verbosity) {
  ConfigurationBase::setup();
}

int Configuration<Spatter::OpenMP>::run(bool timed, unsigned long run_id) {
  omp_set_num_threads(omp_threads);
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::OpenMP>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = sparse.data() + delta * i;
      double *tl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[j]];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

}

void Configuration<Spatter::OpenMP>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = sparse.data() + delta * i;
      double *sl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[j]] = sl[j];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    double *tl = sparse_scatter.data() + delta_scatter * i;
    double *sl = sparse_gather.data() + delta_gather * i;

#pragma omp simd
    for (size_t j = 0; j < pattern_length; ++j) {
      tl[pattern_scatter[j]] = sl[pattern_gather[j]];
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::multi_gather(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = sparse.data() + delta * i;
      double *tl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[pattern_gather[j]]];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}


void Configuration<Spatter::OpenMP>::multi_scatter(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = sparse.data() + delta * i;
      double *sl = dense_perthread[t].data() + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[pattern_scatter[j]]] = sl[j];
      }
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}
#endif

#ifdef USE_CUDA
Configuration<Spatter::CUDA>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread, double *&dev_dense,
    size_t &dense_size, const size_t delta, const size_t delta_gather,
    const size_t delta_scatter, const long int seed, const size_t wrap,
    const size_t count, const size_t shared_mem, const size_t local_work_size,
    const unsigned long nruns, const bool aggregate, const bool atomic,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather, delta_scatter, seed,
          wrap, count, shared_mem, local_work_size, 1, nruns, aggregate, atomic,
          verbosity) {
  
  setup();
}

Configuration<Spatter::CUDA>::~Configuration() {
  checkCudaErrors(cudaFree(dev_pattern));
  checkCudaErrors(cudaFree(dev_pattern_gather));
  checkCudaErrors(cudaFree(dev_pattern_scatter));

  if (dev_sparse) {
    checkCudaErrors(cudaFree(dev_sparse));
    dev_sparse = nullptr;
  }

  if (dev_sparse_gather) {
    checkCudaErrors(cudaFree(dev_sparse_gather));
    dev_sparse_gather = nullptr;
  }

  if (dev_sparse_scatter) {
    checkCudaErrors(cudaFree(dev_sparse_scatter));
    dev_sparse_scatter = nullptr;
  }

  if (dev_dense) {
    checkCudaErrors(cudaFree(dev_dense));
    dev_dense = nullptr;
  }
}

int Configuration<Spatter::CUDA>::run(bool timed, unsigned long run_id) {
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::CUDA>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_gather_wrapper(
      dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = cuda_scatter_atomic_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = cuda_scatter_wrapper(
        dev_pattern, dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms = cuda_scatter_gather_atomic_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);
  else
    time_ms = cuda_scatter_gather_wrapper(dev_pattern_scatter,
        dev_sparse_scatter, dev_pattern_gather, dev_sparse_gather,
        pattern_length, delta_scatter, delta_gather, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_gather(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_gather.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = cuda_multi_gather_wrapper(dev_pattern, dev_pattern_gather,
      dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::multi_scatter(
    bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  float time_ms = 0.0;

  if (atomic)
    time_ms =
        cuda_multi_scatter_atomic_wrapper(dev_pattern, dev_pattern_scatter,
            dev_sparse, dev_dense, pattern_length, delta, wrap, count);
  else
    time_ms = cuda_multi_scatter_wrapper(dev_pattern, dev_pattern_scatter,
        dev_sparse, dev_dense, pattern_length, delta, wrap, count);

  checkCudaErrors(cudaDeviceSynchronize());

  if (timed)
    time_seconds[run_id] = ((double)time_ms / 1000.0);
}

void Configuration<Spatter::CUDA>::setup() {
  ConfigurationBase::setup();

  checkCudaErrors(
      cudaMalloc((void **)&dev_pattern, sizeof(size_t) * pattern.size()));
  checkCudaErrors(cudaMalloc(
      (void **)&dev_pattern_gather, sizeof(size_t) * pattern_gather.size()));
  checkCudaErrors(cudaMalloc(
      (void **)&dev_pattern_scatter, sizeof(size_t) * pattern_scatter.size()));

  checkCudaErrors(cudaMemcpy(dev_pattern, pattern.data(),
      sizeof(size_t) * pattern.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pattern_gather, pattern_gather.data(),
      sizeof(size_t) * pattern_gather.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_pattern_scatter, pattern_scatter.data(),
      sizeof(size_t) * pattern_scatter.size(), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaDeviceSynchronize());
}
#endif

} // namespace Spatter
