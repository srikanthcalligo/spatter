#include <stdio.h>

#include "Configuration.hh"
#include <chrono>
#include <cmath>

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

CBHandle MakeCircularBuffer_Float32(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, tt::CBIndex cb_index, uint32_t num_tiles_per_cb, uint32_t single_tile_size)
{
  uint32_t buf_size = num_tiles_per_cb * single_tile_size * sizeof(float);
  CircularBufferConfig cb0_config = CircularBufferConfig(
        buf_size,
        {{cb_index, tt::DataFormat::Float32}}).set_page_size(cb_index, single_tile_size * sizeof(float));
    
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

std::shared_ptr<tt::tt_metal::Buffer> MakeBuffer_L1(IDevice *device, uint32_t size, uint32_t page_size, uint32_t dtype_size)
{
    InterleavedBufferConfig config{
        .device= device,
        .size = size * dtype_size,
        .page_size = page_size * dtype_size,
        .buffer_type = tt_metal::BufferType::L1
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
template double metalium_gather_wrapper<uint32_t>(const aligned_vector<size_t> &pattern, const aligned_vector<double> &sparse,
    aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device, CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

//Scatter Kernel
template double metalium_scatter_wrapper<uint32_t>(const aligned_vector<size_t> &pattern, aligned_vector<double> &sparse,
    const aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device, CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

//Scatter_gather Kernel
template double metalium_scatter_gather_wrapper<uint32_t>(const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse_scatter, const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<double> &sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

    //multi_gather
template double metalium_multi_gather_wrapper<uint32_t>(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather, const aligned_vector<double> &sparse, aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

//multi_scatter kernel
template double metalium_multi_scatter_wrapper<uint32_t>(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_scatter, aligned_vector<double> &sparse, const aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

//Gather Kernel
template<typename T> 
double metalium_gather_wrapper(const aligned_vector<size_t> &pattern, const aligned_vector<double> &sparse,
    aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle) {
    
    //Divide sparse array into tiles
    uint32_t n_tiles = (sparse.size() / single_tile_size);
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;

    //Initialize device buffers. 
    std::vector<uint32_t> dev_sparse(n_tiles * single_tile_size);
    std::vector<uint32_t> dev_pattern(single_tile_size);
    std::vector<uint32_t> compute_pattern(single_tile_size);

    uint32_t stride = step_size;
    uint32_t iin = 0, icn = 1;
    uint32_t extra_tile = sparse.size() % single_tile_size;

    size_t icn1 = 0, iter = 1, inc = 0, i = 0;
    int flag = 0;
    size_t prev = 0;
    size_t status = 0;
    uint32_t dev_sparse_size = (n_tiles * single_tile_size);
    uint32_t req_tiles = 0;
  
    //Store sparse array as tile based index, so that we can read tile by tile on the device
    //Converting sparse array input datatype to uint32_t/float16, because device will not support double. 
    while(i < count){
      for (size_t j = 0; j < pattern_length; j++) {
        inc = pattern[j] + delta * i;                    
        if((i*delta+(pattern_length - 1) * step_size) >= (iter * single_tile_size)){
                    icn1 = icn1 + (iter * single_tile_size) - (i * delta);
                    iter = iter + 1;
        }        
        if(( inc + icn1 + prev) >= (iter * single_tile_size)){
          flag = 1;
          status = 1;
          prev = icn1 + prev;
          iter = iter + 1;
          break;
        }
      
        if(status){
          status =0;
          prev = prev - ((inc + icn1 + prev) % single_tile_size);
        }
        if( (inc + icn1 + prev) >= dev_sparse_size){
          dev_sparse_size = dev_sparse_size + single_tile_size;
          dev_sparse.resize(dev_sparse_size);
          req_tiles = req_tiles + 1;
        }
        dev_sparse[ inc + icn1 + prev] = static_cast<uint32_t>(sparse[inc]);
      }
      if(flag == 1){
        flag = 0;
      } else{
        i = i + 1;
      } 
    }
    
    n_tiles = n_tiles + req_tiles;

#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif

  
    for (size_t j = 0; j < pattern_length; j++) {
      dev_pattern[j] = static_cast<uint32_t>(pattern[j]);
    }
  
    //Create dram buffers for sparse array
    //Initialize pattern and compute_pattern array in L1 Cache 
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_compute_pattern = MakeBuffer_L1(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense;

    //Write data to the DRAM
    EnqueueWriteBuffer(cq, dram_buffer_sparse, dev_sparse, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, dev_pattern, false);

    if(is_parallel_mode_on){
      //Create a parallel region with the default function
      auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
      uint32_t num_cores_x = compute_with_storage_grid_size.x;
      uint32_t num_cores_y = compute_with_storage_grid_size.y;
      auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
#ifdef PRINT_DEBUG
      std::cout << "No.of Cores : " << num_cores << std::endl;
#endif        
      //Output buffer creation in DRAM
      dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));

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
          if(is_compute_mode_on){
            SetRuntimeArgs(program,
            data_read_kernel_handle,
            core,
            {dram_buffer_sparse->address(),
            dram_buffer_pattern->address(),
            dram_buffer_compute_pattern->address(),
            num_tiles_written,
            num_output_tiles_per_core});

            SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap, is_nr_enabled});
            SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_dense->address(), i, num_cores});
          }else{
            SetRuntimeArgs(program,
            data_read_kernel_handle,
            core,
            {dram_buffer_sparse->address(),
            dram_buffer_pattern->address(),
            n_tiles,
            num_tiles_written,
            num_output_tiles_per_core, 
            i, dram_buffer_dense->address(), pattern_length, delta, extra_tile, stride, single_tile_size, count, num_cores, wrap});
          }
          num_tiles_written += num_output_tiles_per_core;
      }
    } else {
        //Output buffer creation in DRAM
        dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));
        if(is_compute_mode_on){
          SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), dram_buffer_compute_pattern->address(), n_tiles});
          SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap, is_nr_enabled});
          SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_dense->address()}); //Return only the final tile
        } else {
          SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), dram_buffer_dense->address(), n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap});
        }

    }
    
    
    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);

    //Wait here for the device execution to be completed. 
    Finish(cq);

    tt_metal::DumpDeviceProfileResults(device, program);

    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    //Final dense array : Read last tile data from DRAM
    std::vector<uint32_t> dev_dense(single_tile_size);
    EnqueueReadBuffer(cq, dram_buffer_dense, dev_dense, true);

#ifdef PRINT_DEBUG
    int pass = 0;
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
        //if(i == (count - 1)){
        //  printf("%u ", (uint32_t)dense[j + pattern_length * (i % wrap)]);
        //}
      }
    }

    for(uint32_t i= 0; i < pattern_length; i++){
        if(dev_dense[i] != static_cast<uint32_t>(dense[i])){
           pass=1;
           break;
        }
    }
    if(pass == 0){
      printf("\nTest Passed.\n");
    } else {
      printf("\nTest Failed.\n");
    }
    printf("\n");
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}

//Scatter Kernel
template<typename T> 
double metalium_scatter_wrapper(const aligned_vector<size_t> &pattern, aligned_vector<double> &sparse,
    const aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle) {
      
    //Get the tiles count to convert sparse into the tile based index access.
    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;

    size_t icn1 = 0, iter = 1, inc = 0, i = 0;
    int flag = 0;
    size_t prev = 0;
    size_t status = 0;
    uint32_t dev_sparse_size = (n_tiles * single_tile_size);
    uint32_t req_tiles = 0;

    //Convert input double to the required format. 
    std::vector<T> dev_dense(single_tile_size);
    std::vector<T> dev_pattern(single_tile_size);

    uint32_t stride = step_size;
    uint32_t remainder = single_tile_size % pattern_length;
    uint32_t extra_tile = sparse.size() % single_tile_size;

    //Dense array 
    for(int i=0 ; i < dense.size(); i++){
      dev_dense[i] = static_cast<T>(dense[i]);
    }
    //Pattern array
    for(int i=0 ; i < pattern_length ; i++){
      dev_pattern[i] = static_cast<T>(pattern[i]);
    }

    //Get the tiles count to store the final sparse.
    while(i < count){
      for (size_t j = 0; j < pattern_length; j++) {
        inc = pattern[j] + delta * i;                    
        if((i*delta+(pattern_length - 1) * step_size) >= (iter * single_tile_size)){
                    icn1 = icn1 + (iter * single_tile_size) - (i * delta);
                    iter = iter + 1;
        }        
        if(( inc + icn1 + prev) >= (iter * single_tile_size)){
          flag = 1;
          status = 1;
          prev = icn1 + prev;
          iter = iter + 1;
          break;
        }
      
        if(status){
          status =0;
          prev = prev - ((inc + icn1 + prev) % single_tile_size);
        }
        if( (inc + icn1 + prev) >= dev_sparse_size){
          dev_sparse_size = dev_sparse_size + single_tile_size;
          //dev_sparse.resize(dev_sparse_size);
          req_tiles = req_tiles + 1;
        }
        //dev_sparse[ inc + icn1 + prev] = static_cast<uint32_t>(sparse[inc]);
      }
      if(flag == 1){
        flag = 0;
      } else{
        i = i + 1;
      } 
    }
    
    n_tiles = n_tiles + req_tiles;

#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif

    //Create dram buffers for input and output arrays
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse_inter = MakeBuffer_L1(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));

    //Write data to the DRAM
    EnqueueWriteBuffer(cq, dram_buffer_dense, dev_dense, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, dev_pattern, false);

    if(is_parallel_mode_on){
      //Create a parallel region with the default function
      auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
      uint32_t num_cores_x = compute_with_storage_grid_size.x;
      uint32_t num_cores_y = compute_with_storage_grid_size.y;
      auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
#ifdef PRINT_DEBUG
      std::cout << "No.of Cores : " << num_cores << std::endl;
#endif        

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
          if(is_compute_mode_on){
            SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), dram_buffer_sparse_inter->address(), n_tiles, num_tiles_written, num_output_tiles_per_core});
            SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap, num_tiles_written, num_output_tiles_per_core, is_nr_enabled});
            SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse->address(), n_tiles, num_tiles_written, num_output_tiles_per_core});
          }else{
            SetRuntimeArgs(program, data_read_kernel_handle, core,
            {dram_buffer_dense->address(), dram_buffer_pattern->address(), dram_buffer_sparse->address(), n_tiles,pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap,
              num_tiles_written,
              num_output_tiles_per_core, 
              i});
          }
          num_tiles_written += num_output_tiles_per_core;
      }
    } else {

          if(is_compute_mode_on){
            SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), dram_buffer_sparse_inter->address(), n_tiles});
            SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap, is_nr_enabled});
            SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse->address(), n_tiles});
          } else {
            SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), dram_buffer_sparse->address(), n_tiles,pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap});
          }

    }
    
    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);

    //Wait here for the device execution to be completed. 
    Finish(cq);

    //tt_metal::DumpDeviceProfileResults(device, program);

    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    //Final Sparse array
    std::vector<T> dev_sparse(single_tile_size * n_tiles);
    EnqueueReadBuffer(cq, dram_buffer_sparse, dev_sparse, true);

#ifdef PRINT_DEBUG
    std::vector<T> sparse_test(single_tile_size * n_tiles);

    //X86 Run
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
      }
    }

    //TT Test  
    uint32_t loop_count = single_tile_size / delta;
    uint32_t extra_itr = 0;
    uint32_t idx = 0;
    
    if(is_nr_enabled != 1){
      if(pattern_length % delta){
          extra_itr = 1;
      }

      loop_count = loop_count - extra_itr - (stride - 1);
    }

    unsigned long ii = 0;
    for(uint32_t tile_id = 0; tile_id < n_tiles; tile_id++)
    {
        if((tile_id == (n_tiles - 1)) && (extra_tile != 0)){
            loop_count = count - (tile_id * loop_count);
        }

        for(uint32_t i = 0; i < loop_count; i++){
            for(uint32_t j = 0; j < pattern_length; j++){
                sparse_test[dev_pattern[j] + delta * ii] = dev_sparse[(single_tile_size * tile_id) + dev_pattern[j] + delta * i];
            }
            ii++;
        }
    }

    int pass = 0;
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
          if(static_cast<T>(sparse[pattern[j] + delta * i]) != sparse_test[dev_pattern[j] + delta * i]){
            pass = 1;
            break;
          }
      }
      if(pass == 1){
        break;
      }
    }
    if(pass == 0){
      printf("\nTest Passed.\n");
    } else {
      printf("\nTest Failed.\n");
    }
    printf("\n");
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}

//Scatter_Gather Kernel
template<typename T> 
double metalium_scatter_gather_wrapper(const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse_scatter, const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<double> &sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{

    auto compute_tiles = [](size_t total_elems, uint32_t tile_size) -> uint32_t {
        return static_cast<uint32_t>((total_elems + tile_size - 1) / tile_size);
    };

    uint32_t n_tiles_gather = compute_tiles(sparse_gather.size(), single_tile_size);
    uint32_t n_tiles_scatter = compute_tiles(sparse_scatter.size(), single_tile_size);

    //Initialize device buffers. 
    std::vector<uint32_t> dev_sparse_gather(n_tiles_gather * single_tile_size);
    std::vector<uint32_t> dev_pattern_gather(single_tile_size);
    std::vector<uint32_t> dev_pattern_scatter(single_tile_size);
    std::vector<uint32_t> dev_sparse_scatter(n_tiles_scatter * single_tile_size);

    uint32_t stride = pattern_gather[1];
    uint32_t iin = 0, icn = 1;
    uint32_t extra_tile = sparse_gather.size() % single_tile_size;
    uint32_t no_cores = 1;

    size_t icn1 = 0, iter = 1, inc = 0, i = 0;
    int flag = 0;
    size_t prev = 0;
    size_t status = 0;
    uint32_t dev_sparse_gather_size = (n_tiles_gather * single_tile_size);
    uint32_t req_tiles = 0;
  
    //Store sparse_gather array as tile based index, so that we can read tile by tile on the device
    //Converting sparse_gather array input datatype to uint32_t/float16, because device will not support double. 
    while(i < count){
      for (size_t j = 0; j < pattern_length; j++) {
        inc = pattern_gather[j] + delta_gather * i;                    
        if((i*delta_gather+(pattern_length - 1) * stride) >= (iter * single_tile_size)){
                    icn1 = icn1 + (iter * 1024) - (i * delta_gather);
                    iter = iter + 1;
        }        
        if(( inc + icn1 + prev) >= (iter * single_tile_size)){
          flag = 1;
          status = 1;
          prev = icn1 + prev;
          iter = iter + 1;
          break;
        }
      
        if(status){
          status =0;
          prev = prev - ((inc + icn1 + prev) % single_tile_size);
        }
        if( (inc + icn1 + prev) >= dev_sparse_gather_size){
          dev_sparse_gather_size = dev_sparse_gather_size + single_tile_size;
          dev_sparse_gather.resize(dev_sparse_gather_size);
          dev_sparse_scatter.resize(dev_sparse_gather_size);
          req_tiles = req_tiles + 1;
        }
        dev_sparse_gather[ inc + icn1 + prev] = static_cast<uint32_t>(sparse_gather[inc]);
      }
      if(flag == 1){
        flag = 0;
      } else{
        i = i + 1;
      } 
    }
    
    n_tiles_gather = n_tiles_gather + req_tiles;
    n_tiles_scatter = n_tiles_scatter + req_tiles;

#ifdef PRINT_DEBUG
    std::cout << "Tiles gather: " << n_tiles_gather
              << ", Tiles scatter: " << n_tiles_scatter << std::endl;
#endif

    for (size_t j = 0; j < pattern_length; j++) {
      dev_pattern_gather[j] = static_cast<uint32_t>(pattern_gather[j]);
      dev_pattern_scatter[j] = static_cast<uint32_t>(pattern_scatter[j]);
      
#ifdef PRINT_DEBUG_V0
      printf(" %u %u \n", dev_pattern_gather[j],dev_pattern_scatter[j]);
#endif      
    }
  
    //Create dram buffers for sparse array
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse_gather = MakeBuffer(device, single_tile_size * n_tiles_gather, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern_gather = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern_scatter = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse_scatter = MakeBuffer(device, single_tile_size * n_tiles_scatter, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse_inter = MakeBuffer_L1(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    
    //Write data to the DRAM
    EnqueueWriteBuffer(cq, dram_buffer_sparse_gather, dev_sparse_gather, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern_gather, dev_pattern_gather, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern_scatter, dev_pattern_scatter, false);

    if(is_parallel_mode_on){
      //Create a parallel region with the default function
      auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
      uint32_t num_cores_x = compute_with_storage_grid_size.x;
      uint32_t num_cores_y = compute_with_storage_grid_size.y;
      auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles_gather);
#ifdef PRINT_DEBUG
      std::cout << "No.of Cores : " << num_cores << std::endl;
#endif        
      no_cores = num_cores;
      
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
          if(is_compute_mode_on){
            SetRuntimeArgs(program,
              data_read_kernel_handle,
              core,
              {dram_buffer_sparse_gather->address(),
                dram_buffer_pattern_gather->address(),
                dram_buffer_pattern_scatter->address(),
                dram_buffer_sparse_inter->address(),
                n_tiles_gather,
                num_tiles_written,
                num_output_tiles_per_core});

            SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles_gather, pattern_length, delta_gather, delta_scatter, extra_tile, stride, single_tile_size, count, wrap, num_tiles_written, num_output_tiles_per_core});
            SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse_scatter->address(), n_tiles_scatter, num_tiles_written, num_output_tiles_per_core});
          }else{
            printf("TBD\n");
          }
          num_tiles_written += num_output_tiles_per_core;

       }
    } else {
        //Output buffer creation in DRAM
        //  dram_buffer_sparse_scatter = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));
        if(is_compute_mode_on){
          SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_sparse_gather->address(), dram_buffer_pattern_gather->address(),  dram_buffer_pattern_scatter->address(), dram_buffer_sparse_inter->address(), n_tiles_gather});
          SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles_gather, pattern_length, delta_gather, delta_scatter, extra_tile, stride, single_tile_size, count, wrap});
          SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse_scatter->address(), n_tiles_scatter});
        } else {
          printf("TBD\n");
        }

    }
    
    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);

    //Wait here for the device execution to be completed. 
    Finish(cq);

   // tt_metal::DumpDeviceProfileResults(device, program);

    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    //Final sparse_scatter array : Read tiles data from DRAM
    EnqueueReadBuffer(cq, dram_buffer_sparse_scatter, dev_sparse_scatter, true);
  
#ifdef PRINT_DEBUG

    //X86 Run
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
            sparse_gather[pattern_gather[j] + delta_gather * i];
        //printf("%ld  %ld\n", pattern_scatter[j] + delta_scatter * i, pattern_gather[j] + delta_gather * i);
      }
    }

    //TT Test
    std::vector<T> sparse_scatter_test(sparse_scatter.size());
  
    uint32_t loop_count = single_tile_size / delta_gather;
    uint32_t extra_itr = 0;
    uint32_t idx = 0;

    if(is_nr_enabled != 1){
      if(pattern_length % delta_gather){
          extra_itr = 1;
      }

      loop_count = loop_count - extra_itr - (stride - 1);
    }

    uint32_t ii = 0;
    for(uint32_t tile_id = 0; tile_id < n_tiles_gather; tile_id++)
    {
        if((tile_id == (n_tiles_gather - 1)) && (extra_tile != 0)){
            loop_count = count - (tile_id * loop_count);
        }

        for(uint32_t i = 0; i < loop_count; i++){
            for(uint32_t j = 0; j < pattern_length; j++){
                sparse_scatter_test[pattern_scatter[j] + delta_scatter * ii] = dev_sparse_scatter[(single_tile_size * tile_id) + (pattern_gather[j] + delta_gather * i)];
            }
            ii++;
        }
    }

    int pass = 0;
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
          if(static_cast<T>(sparse_scatter[pattern_scatter[j] + delta_scatter * i]) != sparse_scatter_test[pattern_scatter[j] + delta_scatter * i]){
            pass = 1;
            break;
          }
      }
      if(pass == 1){
        break;
      }
    }
    if(pass == 0){
      printf("\nTest Passed.\n");
    } else {
      printf("\nTest Failed.\n");
    }
    printf("\n");
      
    
#endif

  std::chrono::duration<double> time_duration =  end_time - start_time;
  double elapsed_time = time_duration.count();

  return elapsed_time;
}

//Multi_gather Kernel
template<typename T> 
double metalium_multi_gather_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather, const aligned_vector<double> &sparse, aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{    

    //Divide sparse array into tiles
    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;
  
    //Initialize device buffers. 
    std::vector<uint32_t> dev_sparse(n_tiles * single_tile_size);
    std::vector<uint32_t> dev_pattern(single_tile_size);
    std::vector<uint32_t> dev_pattern_gather(single_tile_size);
    std::vector<uint32_t> compute_pattern(single_tile_size);

    uint32_t stride = pattern[pattern_gather[1]];

    uint32_t iin = 0, icn = 1;
    uint32_t extra_tile = sparse.size() % single_tile_size;
    uint32_t no_cores = 1;

    size_t icn1 = 0, iter = 1, inc = 0, i = 0;
    int flag = 0;
    size_t prev = 0;
    size_t status = 0;
    uint32_t dev_sparse_size = (n_tiles * single_tile_size);
    uint32_t req_tiles = 0;

    //Store sparse array as tile based index, so that we can read tile by tile on the device
    //Converting sparse array input datatype to uint32_t/float16, because device will not support double. 
    while(i < count){
      for (size_t j = 0; j < pattern_length; j++) {
        inc = pattern[pattern_gather[j]] + delta * i;                    
        if((i*delta+(pattern_length - 1) * stride) >= (iter * single_tile_size)){
                    icn1 = icn1 + (iter * single_tile_size) - (i * delta);
                    iter = iter + 1;
        }        
        if(( inc + icn1 + prev) >= (iter * single_tile_size)){
          flag = 1;
          status = 1;
          prev = icn1 + prev;
          iter = iter + 1;
          break;
        }
      
        if(status){
          status =0;
          prev = prev - ((inc + icn1 + prev) % single_tile_size);
        }
        if( (inc + icn1 + prev) >= dev_sparse_size){
          dev_sparse_size = dev_sparse_size + single_tile_size;
          dev_sparse.resize(dev_sparse_size);
          req_tiles = req_tiles + 1;
        }
        dev_sparse[ inc + icn1 + prev] = static_cast<uint32_t>(sparse[inc]);
      }
      if(flag == 1){
        flag = 0;
      } else{
        i = i + 1;
      } 
    }

    n_tiles = n_tiles + req_tiles;

#ifdef PRINT_DEBUG
    std::cout << "No.of tiles : " << n_tiles << std::endl;
#endif 

    for (size_t j = 0; j < pattern_length; j++) {
      dev_pattern[j] = static_cast<uint32_t>(pattern[j]);
      dev_pattern_gather[j] = static_cast<uint32_t>(pattern_gather[j]);
    }
    
   
    //Create dram buffers for sparse array
    //Initialize pattern and compute_pattern array in L1 Cache 
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern_gather = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_compute_pattern = MakeBuffer_L1(device, single_tile_size, single_tile_size, sizeof(uint32_t));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));

    //Write data to the DRAM
    EnqueueWriteBuffer(cq, dram_buffer_sparse, dev_sparse, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, dev_pattern, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern_gather, dev_pattern_gather, false);

    if(is_parallel_mode_on){
      //Create a parallel region with the default function
      auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
      uint32_t num_cores_x = compute_with_storage_grid_size.x;
      uint32_t num_cores_y = compute_with_storage_grid_size.y;
      auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
#ifdef PRINT_DEBUG
      std::cout << "No.of Cores : " << num_cores << std::endl;
#endif        
      no_cores = num_cores;
      //Output buffer creation in DRAM
      //dram_buffer_dense = MakeBuffer(device, single_tile_size * num_cores, single_tile_size, sizeof(uint32_t));
      dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(uint32_t));

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

          if(is_compute_mode_on){
            SetRuntimeArgs(program,
            data_read_kernel_handle,
            core,
            {dram_buffer_sparse->address(),
            dram_buffer_pattern->address(),
            dram_buffer_pattern_gather->address(),
            dram_buffer_compute_pattern->address(),
            num_tiles_written,
            num_output_tiles_per_core});

            SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, num_tiles_written, num_output_tiles_per_core, i, num_cores, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap});
            SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_dense->address(), i, num_cores});
          }else{
            SetRuntimeArgs(program,
            data_read_kernel_handle,
            core,
            {dram_buffer_sparse->address(),
            dram_buffer_pattern->address(),
            dram_buffer_pattern_gather->address(),
            n_tiles,
            num_tiles_written,
            num_output_tiles_per_core, 
            i, dram_buffer_dense->address(), pattern_length, delta, extra_tile, stride, single_tile_size, count, num_cores, wrap});
          }
          num_tiles_written += num_output_tiles_per_core;
      }
    }else{
      //Output buffer creation in DRAM
       if(is_compute_mode_on){
          SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), dram_buffer_pattern_gather->address(), dram_buffer_compute_pattern->address(), n_tiles});
          SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap});
          SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_dense->address()}); //Return only the final tile
        } else {
          SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_sparse->address(), dram_buffer_pattern->address(), dram_buffer_pattern_gather->address(), dram_buffer_dense->address(), n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap});
        }
    }

    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    EnqueueProgram(cq, program, false);

    //Wait here for the device execution to be completed. 
    Finish(cq);

    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    //Final dense array : Read last tile data from DRAM
    std::vector<uint32_t> dev_dense(single_tile_size);

    EnqueueReadBuffer(cq, dram_buffer_dense, dev_dense, true);

#ifdef PRINT_DEBUG
    //X86 Run
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        dense[j + pattern_length * (i % wrap)] = sparse[pattern[pattern_gather[j]] + delta * i];
      }
    }
    int pass_count=0, fail_count =0;
    for(uint32_t i= 0; i < pattern_length; i++){
      if(dev_dense[i] == static_cast<uint32_t>(dense[i]))
          pass_count++;
      else
        fail_count++;
    }
     
     if(fail_count)
        printf("\nTest Failed.\n");
      else
        printf("\nTest Passed.\n");
    
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}

//Multi_Scatter Kernel
template<typename T> 
double metalium_multi_scatter_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_scatter, aligned_vector<double> &sparse, const aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{
    
    //Get the tiles count to convert sparse into the tile based index access.
    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;

    size_t icn1 = 0, iter = 1, inc = 0, i = 0;
    int flag = 0;
    size_t prev = 0;
    size_t status = 0;
    uint32_t dev_sparse_size = (n_tiles * single_tile_size);
    uint32_t req_tiles = 0;

    //Convert input double to the required format. 
    std::vector<T> dev_dense(single_tile_size);
    std::vector<T> dev_pattern(single_tile_size);
    std::vector<T> dev_pattern_scatter(single_tile_size);

    uint32_t stride = pattern[1];
    uint32_t remainder = single_tile_size % pattern_length;
    uint32_t extra_tile = sparse.size() % single_tile_size;

    //Dense array 
    for(int i=0 ; i < dense.size(); i++){
      dev_dense[i] = static_cast<T>(dense[i]);
    }
    //Pattern array
    for(int i=0 ; i < pattern_length ; i++){
      dev_pattern[i] = static_cast<T>(pattern[i]);
    }

    for(int i=0 ; i < pattern_scatter.size() ; i++){
      dev_pattern_scatter[i] = static_cast<T>(pattern_scatter[i]);
    }

    //Get the tiles count to store the final sparse.
    while(i < count){
      for (size_t j = 0; j < pattern_length; j++) {
        inc = pattern[j] + delta * i;                    
        if((i*delta+(pattern_length - 1) * pattern[1]) >= (iter * single_tile_size)){
                    icn1 = icn1 + (iter * single_tile_size) - (i * delta);
                    iter = iter + 1;
        }        
        if(( inc + icn1 + prev) >= (iter * single_tile_size)){
          flag = 1;
          status = 1;
          prev = icn1 + prev;
          iter = iter + 1;
          break;
        }
      
        if(status){
          status =0;
          prev = prev - ((inc + icn1 + prev) % single_tile_size);
        }
        if( (inc + icn1 + prev) >= dev_sparse_size){
          dev_sparse_size = dev_sparse_size + single_tile_size;
          req_tiles = req_tiles + 1;
        }
      }
      if(flag == 1){
        flag = 0;
      } else{
        i = i + 1;
      } 
    }
    
    n_tiles = n_tiles + req_tiles;

#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif
    //Create dram buffers for input and output arrays
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse = MakeBuffer(device, single_tile_size * n_tiles, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_sparse_inter = MakeBuffer_L1(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_pattern_scatter = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_dense = MakeBuffer(device, single_tile_size, single_tile_size, sizeof(T));

    //Write data to the DRAM
    EnqueueWriteBuffer(cq, dram_buffer_dense, dev_dense, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern, dev_pattern, false);
    EnqueueWriteBuffer(cq, dram_buffer_pattern_scatter, dev_pattern_scatter, false);

    if(is_parallel_mode_on){
      //Create a parallel region with the default function
      auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
      uint32_t num_cores_x = compute_with_storage_grid_size.x;
      uint32_t num_cores_y = compute_with_storage_grid_size.y;
      auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, n_tiles);
#ifdef PRINT_DEBUG
      std::cout << "No.of Cores : " << num_cores << std::endl;
#endif        

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
          if(is_compute_mode_on){
            SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), dram_buffer_sparse_inter->address(), dram_buffer_pattern_scatter->address(), n_tiles, num_tiles_written, num_output_tiles_per_core});
            SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap, num_tiles_written, num_output_tiles_per_core});
            SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse->address(), n_tiles, num_tiles_written, num_output_tiles_per_core});
          }else{
            SetRuntimeArgs(program, data_read_kernel_handle, core,
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
              remainder, stride, count, dram_buffer_pattern_scatter->address()});
          }
          num_tiles_written += num_output_tiles_per_core;
      }
    } else {

          if(is_compute_mode_on){
            SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), dram_buffer_sparse_inter->address(), dram_buffer_pattern_scatter->address(), n_tiles});
            SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap});
            SetRuntimeArgs(program, data_write_kernel_handle, core, {dram_buffer_sparse->address(), n_tiles});
          } else {
            SetRuntimeArgs(program, data_read_kernel_handle, core, {dram_buffer_dense->address(), dram_buffer_pattern->address(), n_tiles, dram_buffer_sparse->address(), pattern_length, delta, wrap, single_tile_size, extra_tile, stride, count, dram_buffer_pattern_scatter->address()});
          }

    }
    
    //Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);

    //Wait here for the device execution to be completed. 
    Finish(cq);

    //tt_metal::DumpDeviceProfileResults(device, program);

    //End the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    //Final Sparse array
    std::vector<T> dev_sparse(single_tile_size * n_tiles);
    EnqueueReadBuffer(cq, dram_buffer_sparse, dev_sparse, true);

#ifdef PRINT_DEBUG
    //X86 Run
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        sparse[pattern[pattern_scatter[j]] + delta * i] = dense[j + pattern_length * (i % wrap)];
      }
    }

    //TT Test
    std::vector<T> sparse_test(single_tile_size * n_tiles);
  
    uint32_t loop_count = single_tile_size / delta;
    uint32_t extra_itr = 0;
    uint32_t idx = 0;

    if(is_nr_enabled != 1){
      if(pattern_length % delta){
          extra_itr = 1;
      }

      loop_count = loop_count - extra_itr - (stride - 1);
    }
    
    uint32_t ii = 0;
    for(uint32_t tile_id = 0; tile_id < n_tiles; tile_id++)
    {
        if((tile_id == (n_tiles - 1)) && (extra_tile != 0)){
            loop_count = count - (tile_id * loop_count);
        }

        for(uint32_t i = 0; i < loop_count; i++){
            for(uint32_t j = 0; j < pattern_length; j++){
                sparse_test[dev_pattern[dev_pattern_scatter[j]] + delta * ii] = dev_sparse[(single_tile_size * tile_id) + (dev_pattern[dev_pattern_scatter[j]] + delta * i)];
            }
            ii++;
        }
    }
    
    int pass = 0;
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
          if(static_cast<T>(sparse[dev_pattern[dev_pattern_scatter[j]] + delta * i]) != sparse_test[dev_pattern[dev_pattern_scatter[j]] + delta * i]){
            pass = 1;
            break;
          }
      }
      if(pass == 1){
        break;
      }
    }
    if(pass == 0){
      printf("\nTest Passed.\n");
    } else {
      printf("\nTest Failed.\n");
    }
    printf("\n");
#endif

    std::chrono::duration<double> time_duration =  end_time - start_time;
    double elapsed_time = time_duration.count();

    return elapsed_time;
}
