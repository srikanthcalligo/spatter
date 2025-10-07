/*!
  \file Configuration.cc
*/

#include <numeric>
#include <atomic>

#include "Configuration.hh"

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
    const bool atomic, const unsigned long verbosity, size_t tt_compute_mode)
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
      verbosity(verbosity), tt_compute_mode(tt_compute_mode), time_seconds(nruns, 0) {
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

#ifdef USE_TT_METAL

  size_t dtype_size = sizeof(uint32_t);

  if(tt_compute_mode) {
      //dtype_size = sizeof(bfloat16);
      dtype_size = sizeof(uint32_t);
  }
  if (kernel.compare("gather") == 0 || kernel.compare("scatter") == 0)
    bytes_moved = pattern.size() * count * dtype_size;

  if (kernel.compare("sg") == 0)
    bytes_moved = (pattern_scatter.size() + pattern_gather.size()) * count * dtype_size;

  if (kernel.compare("multiscatter") == 0)
    bytes_moved = pattern_scatter.size() * count * dtype_size;

  if (kernel.compare("multigather") == 0)
    bytes_moved = pattern_gather.size() * count * dtype_size;

#else

  if (kernel.compare("gather") == 0 || kernel.compare("scatter") == 0)
    bytes_moved = pattern.size() * count * sizeof(size_t);

  if (kernel.compare("sg") == 0)
    bytes_moved = (pattern_scatter.size() + pattern_gather.size()) * count * sizeof(size_t);

  if (kernel.compare("multiscatter") == 0)
    bytes_moved = pattern_scatter.size() * count * sizeof(size_t);

  if (kernel.compare("multigather") == 0)
    bytes_moved = pattern_gather.size() * count * sizeof(size_t);

#endif

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

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i){
    for (size_t j = 0; j < pattern_length; ++j){
      dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
    }
  }
  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
 if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j)
      sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::scatter_gather(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i){
    for (size_t j = 0; j < pattern_length; ++j){
      sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
          sparse_gather[pattern_gather[j] + delta_gather * i];
    }
  }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::Serial>::multi_gather(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j)
    {
      dense[j + pattern_length * (i % wrap)] =
          sparse[pattern[pattern_gather[j]] + delta * i];
    }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

}

void Configuration<Spatter::Serial>::multi_scatter(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

  for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j){
      sparse[pattern[pattern_scatter[j]] + delta * i] =
          dense[j + pattern_length * (i % wrap)];
    }

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

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

#ifdef USE_TT_METAL
Configuration<Spatter::TT_Metalium>::Configuration(const size_t id,
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
    const bool aggregate, const unsigned long verbosity, size_t tt_compute_mode, size_t tt_parallel_mode, size_t tt_core_id)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather,
          delta_scatter, seed, wrap, count, 0, 1024, 1, nruns, aggregate, false,
          verbosity, tt_compute_mode) {

  is_compute_mode_on = tt_compute_mode;
  is_parallel_mode_on = tt_parallel_mode;
  
  core_id = tt_core_id;
  if(is_parallel_mode_on == 0){
    if((core_id >= 0) && (core_id < 64)){
      core = {core_id / 9, core_id % 9};
      printf("\nCore ID = %zu  Grid_x  = %u Grid_y = %u\n", tt_core_id, core_id / 9, core_id % 9); 
    }else{
      printf("Core id should be less than 64\n");
      exit(0);
    }
  }
  ConfigurationBase::setup();
}

Configuration<Spatter::TT_Metalium>::~Configuration() {
    CloseDevice(device);
}

int Configuration<Spatter::TT_Metalium>::run(bool timed, unsigned long run_id) {
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::TT_Metalium>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();
  //To add CB's and Kernel ID's only one time to the program
  if(is_first_run == 0) {
    setup();
    is_first_run = is_first_run + 1;
  }
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
    
  if(is_compute_mode_on) {
      kernel_exec_time = metalium_gather_wrapper<uint32_t>(pattern, sparse, dense,
                          pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, data_write_kernel_handle, compute_kernel_handle);
  } else {
      kernel_exec_time = metalium_gather_wrapper<uint32_t>(pattern, sparse, dense,
                          pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, 0, 0);
  }

  if (timed)
    time_seconds[run_id] = kernel_exec_time;
}

void Configuration<Spatter::TT_Metalium>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  //To add CB's and Kernel ID's only one time to the program
  if(is_first_run == 0) {
    setup();
    is_first_run = is_first_run + 1;
  }  

  if(is_compute_mode_on) {
      kernel_exec_time = metalium_scatter_wrapper<uint32_t>(pattern, sparse, dense,
                          pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, data_write_kernel_handle, compute_kernel_handle);
  } else {
      kernel_exec_time = metalium_scatter_wrapper<uint32_t>(pattern, sparse, dense,
                          pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, 0, 0);
  }

  if (timed)
    time_seconds[run_id] = kernel_exec_time;
}

void Configuration<Spatter::TT_Metalium>::scatter_gather(bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  
  //To add CB's and Kernel ID's only one time to the program
  if(is_first_run == 0) {
    setup();
    is_first_run = is_first_run + 1;
  }

  if(is_compute_mode_on) {
      kernel_exec_time = metalium_scatter_gather_wrapper<uint32_t>(pattern_scatter,
                          sparse_scatter, pattern_gather, sparse_gather,
                          pattern_length, delta_scatter, delta_gather, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, data_write_kernel_handle, compute_kernel_handle);
      
  } else {
      printf("Not Implemented.....TBD\n");
      /*kernel_exec_time = metalium_scatter_gather_wrapper<uint32_t>(pattern_scatter,
                          sparse_scatter, pattern_gather, sparse_gather,
                          pattern_length, delta_scatter, delta_gather, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, 0, 0);
      */
  }

  if (timed)
    time_seconds[run_id] = kernel_exec_time;
}

void Configuration<Spatter::TT_Metalium>::multi_gather(bool timed, unsigned long run_id) {
  int pattern_length = static_cast<int>(pattern_gather.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  //To add CB's and Kernel ID's only one time to the program
  if(is_first_run == 0) {
    setup();
    is_first_run = is_first_run + 1;
  }

  if(is_compute_mode_on) {
      kernel_exec_time = metalium_multi_gather_wrapper<uint32_t>(pattern, pattern_gather,
                          sparse, dense, pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, data_write_kernel_handle, compute_kernel_handle);
  } else {
      kernel_exec_time = metalium_multi_gather_wrapper<uint32_t>(pattern, pattern_gather,
                          sparse, dense, pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, 0, 0);
  }

  if (timed)
    time_seconds[run_id] = kernel_exec_time;
}

void Configuration<Spatter::TT_Metalium>::multi_scatter(bool timed, unsigned long run_id) {
    int pattern_length = static_cast<int>(pattern_scatter.size());

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  //To add CB's and Kernel ID's only one time to the program
  if(is_first_run == 0) {
    setup();
    is_first_run = is_first_run + 1;
  }

  if(is_compute_mode_on) {
      kernel_exec_time = metalium_multi_scatter_wrapper<uint32_t>(pattern, pattern_scatter,
                          sparse, dense, pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, data_write_kernel_handle, compute_kernel_handle);
  } else {
      printf("Not Implemented.....TBD\n");
      /*kernel_exec_time = metalium_multi_scatter_wrapper<uint32_t>(pattern, pattern_scatter,
                          sparse, dense, pattern_length, delta, wrap, count, is_compute_mode_on, is_parallel_mode_on,
                          core, device_id, device, cq, program, single_tile_size,
                          data_read_kernel_handle, 0, 0);
      */
  }

  if (timed)
    time_seconds[run_id] = kernel_exec_time;
}

void Configuration<Spatter::TT_Metalium>::setup() {
  //ConfigurationBase::setup();
  
  std::string kernel_file_path = "tt_metal/programming_examples/spatter/src/Spatter/kernels/";
  
  if(is_parallel_mode_on){
    uint32_t n_tiles_required = 0;
    if((kernel.compare("gather") == 0) || (kernel.compare("scatter") == 0) || (kernel.compare("multigather") == 0) || (kernel.compare("multiscatter") == 0)) {
      n_tiles_required = (sparse.size()) / single_tile_size;
      n_tiles_required = (sparse.size() % single_tile_size == 0 ) ? n_tiles_required : n_tiles_required + 1;
    }
    if(kernel.compare("sg") == 0) {
      n_tiles_required = (sparse_gather.size()) / single_tile_size;
      n_tiles_required = (sparse_gather.size() % single_tile_size == 0 ) ? n_tiles_required : n_tiles_required + 1;
    }
    core_set = std::get<1>(split_work_to_cores(device->compute_with_storage_grid_size(), n_tiles_required));
  }
  
  //For Compute Engine
  if(is_compute_mode_on) {
    if(kernel.compare("gather") == 0){
      //Sparse circular buffer id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Pattern array CB id  
      cb_pattern = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      cb_intermediate = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);
      //Dense array CB  id
      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_3, num_tiles_per_cb, single_tile_size);
      std::string kernel_read_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/gather_read_kernel_compute_mc.cpp" : "compute/data/gather_read_kernel_compute.cpp");
      std::string kernel_compute_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/core/gather_compute_kernel_mc.cpp" : "compute/core/gather_compute_kernel.cpp");
      std::string kernel_write_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/gather_write_kernel_compute_mc.cpp" : "compute/data/gather_write_kernel_compute.cpp");
      
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_read_file_name);
      //Create write kernel Handler
      data_write_kernel_handle = Make_Write_NOC1_Kernel(core, core_set, is_parallel_mode_on, program, kernel_write_file_name);
      //Create Compute kernel Handler
      compute_kernel_handle = Make_Compute_Core_Kernel(core, core_set, is_parallel_mode_on, program, kernel_compute_file_name);
    }

    else if(kernel.compare("scatter") == 0){
      //Sparse circular buffer id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_3, num_tiles_per_cb, single_tile_size);
      //Pattern array CB id  
      cb_pattern = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //Dense array CB  id
      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Sparse intermediate CB
      cb_intermediate = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);

      std::string kernel_read_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/scatter_read_kernel_compute_mc.cpp" : "compute/data/scatter_read_kernel_compute.cpp");
      std::string kernel_compute_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/core/scatter_compute_kernel_mc.cpp" : "compute/core/scatter_compute_kernel.cpp");
      std::string kernel_write_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/scatter_write_kernel_compute_mc.cpp" : "compute/data/scatter_write_kernel_compute.cpp");
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_read_file_name);
      //Create write kernel Handler
      data_write_kernel_handle = Make_Write_NOC1_Kernel(core, core_set, is_parallel_mode_on, program, kernel_write_file_name);
      //Create Compute kernel Handler
      compute_kernel_handle = Make_Compute_Core_Kernel(core, core_set, is_parallel_mode_on, program, kernel_compute_file_name);
    }

    else if(kernel.compare("multiscatter") == 0){
      //Pattern array CB id  
      cb_pattern = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //Scatter pattern array CB id  
      cb_pattern_scatter = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);
      //sparse array CB  id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_4, num_tiles_per_cb, single_tile_size);
      //dense array CB  id
      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Sparse intermediate CB
      cb_intermediate = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_3, num_tiles_per_cb, single_tile_size);

      std::string kernel_read_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/multi_scatter_read_kernel_compute_mc.cpp" : "compute/data/multi_scatter_read_kernel_compute.cpp");
      std::string kernel_compute_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/core/multi_scatter_compute_kernel_mc.cpp" : "compute/core/multi_scatter_compute_kernel.cpp");
      std::string kernel_write_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/multi_scatter_write_kernel_compute_mc.cpp" : "compute/data/multi_scatter_write_kernel_compute.cpp");
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_read_file_name);
      //Create write kernel Handler
      data_write_kernel_handle = Make_Write_NOC1_Kernel(core, core_set, is_parallel_mode_on, program, kernel_write_file_name);
      //Create Compute kernel Handler
      compute_kernel_handle = Make_Compute_Core_Kernel(core, core_set, is_parallel_mode_on, program, kernel_compute_file_name);
    }

    else if(kernel.compare("sg") == 0){
      //Sparse circular buffer id
      cb_pattern_gather = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Pattern array CB id  
      cb_pattern_scatter = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //Dense array CB  id
      cb_sparse_gather = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);
      cb_sparse_scatter = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_4, num_tiles_per_cb, single_tile_size);
      //Sparse intermediate CB
      cb_intermediate = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_3, num_tiles_per_cb, single_tile_size);

      std::string kernel_read_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/scatter_gather_read_kernel_compute_mc.cpp" : "compute/data/scatter_gather_read_kernel_compute.cpp");
      std::string kernel_compute_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/core/scatter_gather_compute_kernel_mc.cpp" : "compute/core/scatter_gather_compute_kernel.cpp");
      std::string kernel_write_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/scatter_gather_write_kernel_compute_mc.cpp" : "compute/data/scatter_gather_write_kernel_compute.cpp");
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_read_file_name);
      //Create write kernel Handler
      data_write_kernel_handle = Make_Write_NOC1_Kernel(core, core_set, is_parallel_mode_on, program, kernel_write_file_name);
      //Create Compute kernel Handler
      compute_kernel_handle = Make_Compute_Core_Kernel(core, core_set, is_parallel_mode_on, program, kernel_compute_file_name);
    }

    else if(kernel.compare("multigather") == 0){
      //Sparse circular buffer id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Pattern array CB id  
      cb_pattern = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //Pattern_gather array CB id  
      cb_pattern_gather = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);

      cb_intermediate = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_3, num_tiles_per_cb, single_tile_size);
      //Dense array CB  id
      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_4, num_tiles_per_cb, single_tile_size);
      std::string kernel_read_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/multi_gather_read_kernel_compute_mc.cpp" : "compute/data/multi_gather_read_kernel_compute.cpp");
      std::string kernel_compute_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/core/multi_gather_compute_kernel_mc.cpp" : "compute/core/multi_gather_compute_kernel.cpp");
      std::string kernel_write_file_name = kernel_file_path + (is_parallel_mode_on ? "compute/data/multi_gather_write_kernel_compute_mc.cpp" : "compute/data/multi_gather_write_kernel_compute.cpp");
      
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_read_file_name);
      //Create write kernel Handler
      data_write_kernel_handle = Make_Write_NOC1_Kernel(core, core_set, is_parallel_mode_on, program, kernel_write_file_name);
      //Create Compute kernel Handler
      compute_kernel_handle = Make_Compute_Core_Kernel(core, core_set, is_parallel_mode_on, program, kernel_compute_file_name);
    }
    else{
      printf("Kernel Not Found\n");
    }

  } else { //For Riscv Mode
    if(kernel.compare("gather") == 0){
      //Sparse circular buffer id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Pattern array CB id  
      cb_pattern = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //Dense array CB  id
      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);

      std::string kernel_file_name = kernel_file_path + (is_parallel_mode_on ? "riscv/gather_kernel_in_riscv_multicore.cpp" : "riscv/gather_kernel_in_riscv.cpp");
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_file_name);
    }

    else if(kernel.compare("scatter") == 0){
      //Sparse circular buffer id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);
      //Pattern array CB id  
      cb_pattern = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //Dense array CB  id
      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      std::string kernel_file_name = kernel_file_path + (is_parallel_mode_on ? "riscv/scatter_kernel_in_riscv_multicore.cpp" : "riscv/scatter_kernel_in_riscv.cpp");
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_file_name);  
    }

    else if(kernel.compare("multigather") == 0){
      //Sparse circular buffer id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Pattern array CB id  
      cb_pattern = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //Dense array CB  id
      cb_pattern_gather = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);

      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_3, num_tiles_per_cb, single_tile_size);

      std::string kernel_file_name = kernel_file_path + (is_parallel_mode_on ? "riscv/multigather_kernel_in_riscv_multicore.cpp" : "riscv/multigather_kernel_in_riscv.cpp");
      //Create read kernel Handler
      data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_file_name);
    }
    else if((kernel.compare("sg") == 0) || (kernel.compare("multiscatter") == 0)){
      //gather pattern circular buffer id
      cb_pattern_gather = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_0, num_tiles_per_cb, single_tile_size);
      //Scatter pattern array CB id  
      cb_pattern_scatter = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_1, num_tiles_per_cb, single_tile_size);
      //sparse array CB  id
      cb_sparse = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_2, num_tiles_per_cb, single_tile_size);
      //dense array CB  id
      cb_dense = MakeCircularBuffer_UInt32(core, core_set, is_parallel_mode_on, program, tt::CBIndex::c_3, num_tiles_per_cb, single_tile_size);

      //Create read kernel Handler
      if (kernel.compare("sg") == 0){
        std::string kernel_file_name = kernel_file_path + (is_parallel_mode_on ? "riscv/scatter_gather_kernel_in_riscv_multicore.cpp" : "riscv/scatter_gather_kernel_in_riscv.cpp");
        data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_file_name);
      }
      
      if (kernel.compare("multiscatter") == 0){
        std::string kernel_file_name = kernel_file_path + (is_parallel_mode_on ? "riscv/multi_scatter_kernel_in_riscv_multicore.cpp" : "riscv/multi_scatter_kernel_in_riscv.cpp");
        data_read_kernel_handle = Make_Read_NOC0_Kernel(core, core_set, is_parallel_mode_on, program, kernel_file_name);  
      }
    } else {
      printf("Kernel Not Found\n");
    }
   
  }
}

#endif //USE_TT_METAL

} // namespace Spatter
