#include <stdio.h>

#include "Configuration.hh"
#include <chrono>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

CBHandle MakeCircularBuffer_UInt32(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, tt::CBIndex cb_index, uint32_t num_tiles_per_cb, uint32_t single_tile_size)
{
  uint32_t buf_size = num_tiles_per_cb * single_tile_size * sizeof(uint32_t);
  CircularBufferConfig cb0_config = CircularBufferConfig(
        buf_size,
        {{cb_index, tt::DataFormat::UInt32}}).set_page_size(cb_index, single_tile_size * sizeof(uint32_t));
    
  if(is_parallel_mode_on){
    return tt_metal::CreateCircularBuffer(
        program,
        core_set,
        cb0_config);
  } else {
    return tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_config);
  }
}

CBHandle MakeCircularBuffer_BFloat16(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, tt::CBIndex cb_index, uint32_t num_tiles_per_cb, uint32_t single_tile_size)
{
  uint32_t buf_size = num_tiles_per_cb * single_tile_size * sizeof(bfloat16);
  CircularBufferConfig cb0_config = CircularBufferConfig(
        buf_size,
        {{cb_index, tt::DataFormat::Float16_b}}).set_page_size(cb_index, single_tile_size * sizeof(bfloat16));
    
  if(is_parallel_mode_on){
    return tt_metal::CreateCircularBuffer(
        program,
        core_set,
        cb0_config);
  } else {
    return tt_metal::CreateCircularBuffer(
        program,
        core,
        cb0_config);
  }
}

KernelHandle Make_Read_NOC0_Kernel(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, std::string kernel_file)
{
  if(is_parallel_mode_on){
    return CreateKernel(
              program,
              kernel_file,
              core_set,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
  } else {
    return CreateKernel(
              program,
              kernel_file,
              core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
  }
}

KernelHandle Make_Write_NOC1_Kernel(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, std::string kernel_file)
{
  if(is_parallel_mode_on){
    return CreateKernel(
              program,
              kernel_file,
              core_set,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
  } else {
    return CreateKernel(
              program,
              kernel_file,
              core,
              DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
  }
}

KernelHandle Make_Compute_Core_Kernel(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, std::string kernel_file, bool f32_dest_acc_en, bool math_approx_mode)
{
  std::vector<uint32_t> compute_kernel_args = {};

  if(is_parallel_mode_on){
    return  CreateKernel(
            program,
            kernel_file,
            core_set,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = f32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
            }
          );
  } else {
    return  CreateKernel(
            program,
            kernel_file,
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = f32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
            }
          );
  }
      
}

std::shared_ptr<tt::tt_metal::Buffer> MakeBuffer(IDevice *device, uint32_t size, uint32_t page_size, uint32_t dtype_size)
{
    InterleavedBufferConfig config{
        .device= device,
        .size = size * dtype_size,
        .page_size = page_size * dtype_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    return CreateBuffer(config);
}

std::shared_ptr<tt::tt_metal::Buffer> MakeBuffer_BFloat16(IDevice *device, uint32_t size, uint32_t page_size)
{
    InterleavedBufferConfig config{
        .device= device,
        .size = size * sizeof(bfloat16),
        .page_size = page_size * sizeof(bfloat16),
        .buffer_type = tt_metal::BufferType::DRAM
    };
    return CreateBuffer(config);
}

//Gather Kernel
template double metalium_gather_wrapper<bfloat16>(const aligned_vector<size_t> &pattern, const aligned_vector<double> &sparse,
    aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device, CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template double metalium_gather_wrapper<uint32_t>(const aligned_vector<size_t> &pattern, const aligned_vector<double> &sparse,
    aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device, CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

//Scatter Kernel
template double metalium_scatter_wrapper<bfloat16>(const aligned_vector<size_t> &pattern, aligned_vector<double> &sparse,
    const aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device, CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template double metalium_scatter_wrapper<uint32_t>(const aligned_vector<size_t> &pattern, aligned_vector<double> &sparse,
    const aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device, CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

//Scatter_gather Kernel
template double metalium_scatter_gather_wrapper<uint32_t>(const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse_scatter, const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<double> &sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template double metalium_scatter_gather_wrapper<bfloat16>(const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse_scatter, const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<double> &sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

//multi_gather
template double metalium_multi_gather_wrapper<uint32_t>(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather, const aligned_vector<double> &sparse, aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template double metalium_multi_scatter_wrapper<uint32_t>(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_scatter, aligned_vector<double> &sparse, const aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);


template<typename T> 
double metalium_gather_wrapper(const aligned_vector<size_t> &pattern, const aligned_vector<double> &sparse,
    aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle) {
    
    //Divide sparse into tiles
    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;

#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif

    //Convert input double to the required format. 
    std::vector<T> dev_sparse(n_tiles * single_tile_size);
    std::vector<T> dev_pattern(single_tile_size);

    uint32_t stride = pattern[1];
    uint32_t remainder = single_tile_size % pattern_length;
    uint32_t iin = 0, icn = 1;
    uint32_t extra_tile = sparse.size() % single_tile_size;
    uint32_t no_cores = 1;
    if constexpr (std::is_integral_v<T>){
      //Sparse array 
      for(int i=0 ; i < sparse.size(); i++){
        if((extra_tile != 0) && (i > 0) && (i % (icn * single_tile_size - remainder) == 0)){
          iin = remainder + (delta - remainder);
          icn = icn + 1;
        }
        dev_sparse[i + iin] = static_cast<T>(sparse[i]);
      }
      //pattern array
      for(int i=0 ; i < pattern_length ; i++){
        dev_pattern[i] = static_cast<T>(pattern[i]);
      }
    }
    else if constexpr (std::is_same_v<T, bfloat16>) {
      for(uint32_t i=0; i < sparse.size() ; i++){
        if((extra_tile != 0) && (i > 0) && (i % (icn * single_tile_size - remainder) == 0)){
          iin = remainder + (delta - remainder);
          icn = icn + 1;
        }
        dev_sparse[i] = bfloat16(static_cast<float>(sparse[i]));
      }
      float in_val = 0;
      for(uint32_t i=0; i < single_tile_size ; i++){
        dev_pattern[i] = bfloat16(static_cast<float>(in_val));
      }
    } else {
      //For any other types
    }

    //Create dram buffers for input and output arrays
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense;

    //Write data to the DRAM
    EnqueueWriteBuffer(cq, dram_buffer_sparse, dev_sparse, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, dev_pattern, false);

    //Pass parameters to the kernel
    if constexpr (std::is_same_v<T, bfloat16>) {
      
      if(is_parallel_mode_on){
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
#ifdef PRINT_DEBUG
        std::cout << "No.of Cores : " << num_cores << std::endl;
#endif        
        no_cores = num_cores;
        //Output buffer creation
        dram_buffer_dense = MakeBuffer(device, single_tile_size * num_cores, single_tile_size, sizeof(T));
        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
          CoreCoord core = {i / num_cores_y, i % num_cores_y};

          uint32_t num_output_tiles_per_core = 0;
          if (core_group_1.contains(core)) {
          num_output_tiles_per_core = num_output_tiles_per_core_group_1;
          } else if (core_group_2.contains(core)) {
              num_output_tiles_per_core = num_output_tiles_per_core_group_2;
          } else {
              TT_ASSERT(false, "Core not in specified core ranges");
          }
          SetRuntimeArgs(program,
          data_read_kernel_handle,
          core,
          {dram_buffer_sparse->address(),
          dram_buffer_pattern->address(),
          n_tiles,
          num_tiles_written,
          num_output_tiles_per_core, 
          i});

          SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
          SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_dense->address(), n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
          num_tiles_written += num_output_tiles_per_core;
        }
      } else {
        //Output buffer creation
        dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
        SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), n_tiles});
        SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles});
        SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_dense->address(), 1}); //Return only the final tile
      }
    } else if constexpr (std::is_integral_v<T>) {
      if(is_parallel_mode_on){
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
#ifdef PRINT_DEBUG
        std::cout << "No.of Cores : " << num_cores << std::endl;
#endif        
        no_cores = num_cores;
        //Output buffer creation
        dram_buffer_dense = MakeBuffer(device, single_tile_size * num_cores, single_tile_size, sizeof(T));
        //Create a parallel region
        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core_id = {i / num_cores_y, i % num_cores_y};

            uint32_t num_output_tiles_per_core = 0;
            if (core_group_1.contains(core_id)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
            } else if (core_group_2.contains(core_id)) {
                num_output_tiles_per_core = num_output_tiles_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }
            SetRuntimeArgs(program,
              data_read_kernel_handle,
              core_id,
              {dram_buffer_sparse->address(),
              dram_buffer_pattern->address(),
              n_tiles,
              dram_buffer_dense->address(),
              pattern_length,
              delta,
              wrap,
              num_tiles_written,
              num_output_tiles_per_core, 
              i,
              single_tile_size,
              extra_tile, stride});
            num_tiles_written += num_output_tiles_per_core;
        }

      }else{
        //Output buffer creation
        dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
        SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), n_tiles, dram_buffer_dense->address(), pattern_length, delta, wrap, single_tile_size, extra_tile, stride});
      }
    }

    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);

    Finish(cq);
    
    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    //Final dense array
    std::vector<T> dev_dense(single_tile_size);
    EnqueueReadBuffer(cq, dram_buffer_dense, dev_dense, true);

#ifdef PRINT_DEBUG
    printf("TT Result : \n");
    if constexpr (std::is_same_v<T, bfloat16>) {
        for(uint32_t i= 0; i < pattern_length; i = i + stride){
          printf("%f ", dev_dense[i].to_float());
        }
    }else if constexpr (std::is_integral_v<T>) {
      if(is_parallel_mode_on){
        for(uint32_t i= (no_cores -1) * single_tile_size; i < (no_cores - 1) * single_tile_size  +  pattern_length; i++){
          printf("%u ", dev_dense[i]);
        }
      } else {
        for(uint32_t i=0 ; i < pattern_length; i++){
          printf("%u ", dev_dense[i]);
        }
      }
    }
    printf("\n\n");
    printf("Expected Result : \n");
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
        if(i == (count - 1)){
          printf("%f ", dense[j + pattern_length * (i % wrap)]);
        }
      }
    }
    printf("\n");
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}

template<typename T> 
double metalium_scatter_wrapper(const aligned_vector<size_t> &pattern, aligned_vector<double> &sparse,
    const aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle) {
    
    //Divide sparse into tiles
    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;

#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif

    //Convert input double to the required format. 
    std::vector<T> dev_dense(single_tile_size);
    std::vector<T> dev_pattern(single_tile_size);

    uint32_t stride = pattern[1];
    uint32_t remainder = single_tile_size % pattern_length;
    uint32_t extra_tile = sparse.size() % single_tile_size;
    
    if constexpr (std::is_integral_v<T>){
      //Sparse array 
      for(int i=0 ; i < dense.size(); i++){
        dev_dense[i] = static_cast<T>(dense[i]);
      }
      //pattern array
      for(int i=0 ; i < pattern_length ; i++){
        dev_pattern[i] = static_cast<T>(pattern[i]);
      }
    }
    else if constexpr (std::is_same_v<T, bfloat16>) {
      for(uint32_t i=0; i < single_tile_size ; i++){
        dev_dense[i] = bfloat16((float)dense[i % pattern_length]);
      }
      float in_val = 0;
      for(uint32_t i=0; i < single_tile_size ; i++){
        dev_pattern[i] = bfloat16(static_cast<float>(in_val));
      }
    } else {
      //For any other types
    }

    //Create dram buffers for input and output arrays
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));

    //Write data to the DRAM
    EnqueueWriteBuffer(cq, dram_buffer_dense, dev_dense, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, dev_pattern, false);

    //Pass parameters to the kernel
    if constexpr (std::is_same_v<T, bfloat16>) {
      if(is_parallel_mode_on){
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
  
          CoreCoord core = {i / num_cores_y, i % num_cores_y};

          uint32_t num_output_tiles_per_core = 0;
          if (core_group_1.contains(core)) {
          num_output_tiles_per_core = num_output_tiles_per_core_group_1;
          } else if (core_group_2.contains(core)) {
              num_output_tiles_per_core = num_output_tiles_per_core_group_2;
          } else {
              TT_ASSERT(false, "Core not in specified core ranges");
          }
          SetRuntimeArgs(program,
          data_read_kernel_handle,
          core,
          {dram_buffer_dense->address(),
          dram_buffer_pattern->address(),
          n_tiles,
          num_tiles_written,
          num_output_tiles_per_core, 
          i});

          SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
          SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse->address(), n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores});
          num_tiles_written += num_output_tiles_per_core;
        }
      } else {
        SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), n_tiles});
        SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles});
        SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse->address(), n_tiles});
      }
    } else if constexpr (std::is_integral_v<T>) {
      if(is_parallel_mode_on){
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
          CoreCoord core = {i / num_cores_y, i % num_cores_y};

          uint32_t num_output_tiles_per_core = 0;
          if (core_group_1.contains(core)) {
          num_output_tiles_per_core = num_output_tiles_per_core_group_1;
          } else if (core_group_2.contains(core)) {
              num_output_tiles_per_core = num_output_tiles_per_core_group_2;
          } else {
              TT_ASSERT(false, "Core not in specified core ranges");
          }
          SetRuntimeArgs(program,
          data_read_kernel_handle,
          core,
          {dram_buffer_dense->address(),
          dram_buffer_pattern->address(),
          n_tiles,
          dram_buffer_sparse->address(),
          pattern_length,
          delta,
          wrap,
          num_tiles_written,
          num_output_tiles_per_core, 
          i,
          single_tile_size,
          remainder, stride});
          num_tiles_written += num_output_tiles_per_core;
        }
      } else {
         SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), n_tiles, dram_buffer_sparse->address(), pattern_length, delta, wrap, single_tile_size, extra_tile, stride});
      }
    }

    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);

    Finish(cq);
    
    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    //Final dense array
    std::vector<T> dev_sparse(single_tile_size * n_tiles);
    EnqueueReadBuffer(cq, dram_buffer_sparse, dev_sparse, true);

#ifdef PRINT_DEBUG
    printf("TT Result : \n");
    if constexpr (std::is_same_v<T, bfloat16>) {
      for(uint32_t i= (n_tiles-1) * single_tile_size; i < (n_tiles - 1) * single_tile_size + pattern_length; i = i+stride){
        printf("%f ", dev_sparse[i].to_float());
      }
    }else if constexpr (std::is_integral_v<T>) {
      for(int i= (n_tiles - 1) * single_tile_size ; i < (n_tiles - 1) * single_tile_size + (pattern_length * pattern[1]); i = i +  pattern[1]){
      printf("%u ", dev_sparse[i]);
    }
    }
    printf("\n\n");
    printf("Expected Result : \n");
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
        if(i == (count -1))
        {
          printf("%f ", sparse[pattern[j] + delta * i]);
        }
      }
    }
    printf("\n");
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}

template<typename T> 
double metalium_scatter_gather_wrapper(const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse_scatter, const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<double> &sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{
    uint32_t n_tiles = (sparse_gather.size()) / single_tile_size;
    uint32_t remainder = sparse_gather.size() % single_tile_size;
    n_tiles = (sparse_gather.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;
#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif
    
    //Input arrarys
    std::vector<T> pattern_scatter_dram(single_tile_size);
    std::vector<T> pattern_gather_dram(single_tile_size);
    std::vector<T> sparse_gather_dram(single_tile_size * n_tiles);
    std::vector<T> sparse_scatter_dram(single_tile_size * n_tiles);

    if constexpr (std::is_same_v<T, bfloat16>) {
      for(int i=0 ; i < sparse_gather.size(); i++){
        sparse_gather_dram[i] = bfloat16(static_cast<float>(sparse_gather[i]));
      }
      for(int i=0 ; i < single_tile_size ; i++){
        pattern_scatter_dram[i] = bfloat16(static_cast<float>(0));
        pattern_gather_dram[i] = bfloat16(static_cast<float>(0));
      }
    }else if constexpr (std::is_integral_v<T>){

      for(int i=0 ; i < sparse_gather.size(); i++){
        sparse_gather_dram[i] = sparse_gather[i];
      }
      for(int i=0 ; i < pattern_length ; i++){
        pattern_scatter_dram[i] = pattern_scatter[i];
        pattern_gather_dram[i] = pattern_gather[i];
      }
    }
    
    std::shared_ptr<tt::tt_metal::Buffer> pattern_scatter_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> pattern_gather_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> sparse_scatter_dram_buffer = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> sparse_gather_dram_buffer = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));

    EnqueueWriteBuffer(cq, pattern_gather_dram_buffer, pattern_gather_dram, false);
    EnqueueWriteBuffer(cq, pattern_scatter_dram_buffer, pattern_scatter_dram, false);
    EnqueueWriteBuffer(cq, sparse_gather_dram_buffer, sparse_gather_dram, false);

    if constexpr (std::is_integral_v<T>){
      if(is_parallel_mode_on){
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
          CoreCoord core = {i / num_cores_y, i % num_cores_y};

          uint32_t num_output_tiles_per_core = 0;
          if (core_group_1.contains(core)) {
          num_output_tiles_per_core = num_output_tiles_per_core_group_1;
          } else if (core_group_2.contains(core)) {
              num_output_tiles_per_core = num_output_tiles_per_core_group_2;
          } else {
              TT_ASSERT(false, "Core not in specified core ranges");
          }
          SetRuntimeArgs(program,
          data_read_kernel_handle,
          core,
          {pattern_gather_dram_buffer->address(),
            pattern_scatter_dram_buffer->address(),
            sparse_gather_dram_buffer->address(),
            sparse_scatter_dram_buffer->address(),
            n_tiles,pattern_length, delta_gather, delta_scatter, count, single_tile_size, remainder,
            num_tiles_written, num_output_tiles_per_core, i
          });
          num_tiles_written += num_output_tiles_per_core;
        }
      } else {
        SetRuntimeArgs(
          program,
          data_read_kernel_handle,
          core, {pattern_gather_dram_buffer->address(),
                pattern_scatter_dram_buffer->address(),
                sparse_gather_dram_buffer->address(),
                sparse_scatter_dram_buffer->address(),
                n_tiles,pattern_length, delta_gather, delta_scatter, count, single_tile_size, remainder});
      }
    } else if constexpr (std::is_same_v<T, bfloat16>){
        SetRuntimeArgs(program, data_read_kernel_handle, core, {pattern_gather_dram_buffer->address(), pattern_scatter_dram_buffer->address(), sparse_gather_dram_buffer->address(), n_tiles});
        SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles});
        SetRuntimeArgs(program, data_write_kernel_handle, core, {sparse_scatter_dram_buffer->address(), n_tiles});
    }
    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();
    
    EnqueueReadBuffer(cq, sparse_scatter_dram_buffer, sparse_scatter_dram, true);

#ifdef PRINT_DEBUG
    printf("Destination array size = %zu %zu %zu %zu %zu\n", sparse_scatter.size(), sparse_gather.size(), pattern_scatter[0], pattern_gather[0], pattern_length);
    printf("Results : \n");
    //Host Run
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
          sparse_gather[pattern_gather[j] + delta_gather * i];    
      }
    }
    if constexpr (std::is_same_v<T, bfloat16>) {
      for(int i= (n_tiles-1) * single_tile_size + pattern_scatter[0]; i < (n_tiles -1) * single_tile_size + 5 * delta_scatter; i = i + delta_scatter){
        printf("TT : %f  Expected : %f\n", sparse_scatter_dram[i].to_float(), sparse_scatter[i]);
      }
    } else if constexpr (std::is_integral_v<T>) {
      for(int i= (n_tiles-1) * single_tile_size + pattern_scatter[0]; i < (n_tiles -1) * single_tile_size + 5 * delta_scatter; i = i + delta_scatter){
        printf("TT : %u  Expected : %f\n", sparse_scatter_dram[i], sparse_scatter[i]);
      }
    }
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}

template<typename T> 
double metalium_multi_gather_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather, const aligned_vector<double> &sparse, aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{    

    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    uint32_t remainder = sparse.size() % single_tile_size;
    uint32_t stride = pattern_gather[pattern_length - 1];
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;
    uint32_t no_cores  = 1;

#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif

    std::shared_ptr<tt::tt_metal::Buffer> pattern_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> pattern_gather_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> sparse_dram_buffer = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dense_dram_buffer;

    //Input pattern and sparse arrary
    std::vector<T> pattern_dram(pattern.size());
    std::vector<T> pattern_gather_dram(pattern_gather.size());
    std::vector<T> sparse_dram(n_tiles * single_tile_size);
    std::vector<T> dense_dram(dense.size());

    uint32_t iin = 0, icn = 1;
    uint32_t tile_remainder = single_tile_size % delta;

    for(int i=0 ; i < sparse.size(); i++){
        if((remainder != 0) && (i > 0) && (i % (icn * single_tile_size - tile_remainder) == 0)){
          iin = tile_remainder + (delta - tile_remainder);
          icn = icn + 1;
        }
        sparse_dram[i + iin] = sparse[i];
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
    if(is_parallel_mode_on){
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
        no_cores = num_cores;
        //std::cout << num_output_tiles_per_core_group_1 << std::endl;
        //std::cout << num_output_tiles_per_core_group_2 << std::endl;

#ifdef PRINT_DEBUG
        std::cout << "No.of Cores : " << n_tiles << std::endl;
#endif        
        //Output buffer creation
        dense_dram_buffer = MakeBuffer(device, single_tile_size * num_cores, single_tile_size, sizeof(T));
        //Create a parallel region
        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core_id = {i / num_cores_y, i % num_cores_y};

            uint32_t num_output_tiles_per_core = 0;
            if (core_group_1.contains(core_id)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
            } else if (core_group_2.contains(core_id)) {
                num_output_tiles_per_core = num_output_tiles_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }
            SetRuntimeArgs(program,
              data_read_kernel_handle,
              core_id,
              {pattern_gather_dram_buffer->address(),
              pattern_dram_buffer->address(),
              sparse_dram_buffer->address(),
              dense_dram_buffer->address(),
              n_tiles,pattern_length, delta, wrap, single_tile_size, remainder, stride,
              num_tiles_written,
              num_output_tiles_per_core, 
              i, count,
              });
            num_tiles_written += num_output_tiles_per_core;
        }

      }else{
        dense_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
        SetRuntimeArgs(
          program,
          data_read_kernel_handle,
          core, {pattern_gather_dram_buffer->address(),
                pattern_dram_buffer->address(),
                sparse_dram_buffer->address(),
                dense_dram_buffer->address(),
                n_tiles,pattern_length, delta, wrap, remainder, single_tile_size, stride, count});
      }
    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //Start the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    EnqueueReadBuffer(cq, dense_dram_buffer, dense_dram, true);

#ifdef PRINT_DEBUG
    //printf("Destination array size = %zu %zu %zu %u %zu\n", sparse.size(), dense.size(), pattern_length, stride, count);

    printf("Results : \n");
    //Host run
    for (size_t i = 0; i < count; ++i)
    for (size_t j = 0; j < pattern_length; ++j)
    {
      dense[j + pattern_length * (i % wrap)] =
          sparse[pattern[pattern_gather[j]] + delta * i];
      if(i == (count - 1)){
        printf("Expected : %f\n", dense[j + pattern_length * (i % wrap)]);
      }
    }
    if(is_parallel_mode_on) {
      for(int i= 0 ; i < no_cores ; i++){
        printf("TT :%u ", dense_dram[i * single_tile_size]);
      }
    }
    else {
      for(int i=0 ; i < pattern_length ; i++){
        printf("TT :%u ", dense_dram[i]);
      }
    }
    printf("\n\n");
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;

}

template<typename T> 
double metalium_multi_scatter_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_scatter, aligned_vector<double> &sparse, const aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{
    
    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    uint32_t remainder = sparse.size() % single_tile_size;
    
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;
    
    uint32_t stride = pattern_scatter[pattern_length - 1];

#ifdef PRINT_DEBUG
    printf("No.of Tiles = %u\n", n_tiles);
#endif
    std::shared_ptr<tt::tt_metal::Buffer> pattern_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> pattern_scatter_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> sparse_dram_buffer = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dense_dram_buffer = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));

    std::vector<T> pattern_dram(pattern.size());
    std::vector<T> pattern_scatter_dram(pattern_scatter.size());
    std::vector<T> sparse_dram(single_tile_size * n_tiles);
    std::vector<T> dense_dram(dense.size());

    for(int i=0 ; i < dense.size(); i++){
      dense_dram[i] = dense[i];
    }
    for(int i=0 ; i < pattern.size() ; i++){
      pattern_dram[i] = pattern[i];
    }

    for(int i=0 ; i < pattern_scatter.size() ; i++){
      pattern_scatter_dram[i] = pattern_scatter[i];
    }

    EnqueueWriteBuffer(cq, pattern_dram_buffer, pattern_dram, false);
    EnqueueWriteBuffer(cq, pattern_scatter_dram_buffer, pattern_scatter_dram, false);
    EnqueueWriteBuffer(cq, dense_dram_buffer, dense_dram, false);
    if(is_parallel_mode_on){
        
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
        
        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
 
          CoreCoord core = {i / num_cores_y, i % num_cores_y};

          uint32_t num_output_tiles_per_core = 0;
          if (core_group_1.contains(core)) {
          num_output_tiles_per_core = num_output_tiles_per_core_group_1;
          } else if (core_group_2.contains(core)) {
              num_output_tiles_per_core = num_output_tiles_per_core_group_2;
          } else {
              TT_ASSERT(false, "Core not in specified core ranges");
          }
          SetRuntimeArgs(program,
          data_read_kernel_handle,
          core,
          {pattern_scatter_dram_buffer->address(),
          pattern_dram_buffer->address(),
          sparse_dram_buffer->address(),
          dense_dram_buffer->address(),
          n_tiles,pattern_length, delta, wrap, single_tile_size, remainder,stride,
          num_tiles_written,
          num_output_tiles_per_core, 
          i
          });
          num_tiles_written += num_output_tiles_per_core;
        }
      } else {
          SetRuntimeArgs(
            program,
            data_read_kernel_handle,
            core, {pattern_scatter_dram_buffer->address(),
                  pattern_dram_buffer->address(),
                  sparse_dram_buffer->address(),
                  dense_dram_buffer->address(),
                  n_tiles,pattern_length, delta, wrap, single_tile_size, remainder});
      }
    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    EnqueueProgram(cq, program, false);

    Finish(cq);

    //Start the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    EnqueueReadBuffer(cq, sparse_dram_buffer, sparse_dram, true);
#ifdef PRINT_DEBUG
    printf("Results : \n");
    for(int i=(n_tiles-1)*single_tile_size + pattern[pattern_scatter[0]]; i < (n_tiles-1)*single_tile_size + pattern[pattern_scatter[0]] + 1; i++){
      printf("TT Output: %u\n", sparse_dram[i]);
    }
    //Host Run
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        sparse[pattern[pattern_scatter[j]] + delta * i] =
            dense[j + pattern_length * (i % wrap)];      
        if(i > count - 1){
          printf("Expected : %f\n", sparse[pattern[pattern_scatter[j]] + delta * i]);
        }
      }
    }
#endif
    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}