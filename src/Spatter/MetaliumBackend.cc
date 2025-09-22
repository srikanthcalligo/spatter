#include <stdio.h>

#include "Configuration.hh"
#include <chrono>

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

//Gather Kernel
template<typename T> 
double metalium_gather_wrapper(const aligned_vector<size_t> &pattern, const aligned_vector<double> &sparse,
    aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle) {
    
    //Divide sparse array into tiles
    uint32_t n_tiles = (sparse.size()) / single_tile_size;
    n_tiles = (sparse.size() % single_tile_size == 0 ) ? n_tiles : n_tiles + 1;

#ifdef PRINT_DEBUG
    std::cout << "No.of Tiles : " << n_tiles << std::endl;
#endif

    //Initialize device buffers. 
    std::vector<uint32_t> dev_sparse(n_tiles * single_tile_size);
    std::vector<uint32_t> dev_pattern(single_tile_size);
    std::vector<uint32_t> compute_pattern(single_tile_size);

    uint32_t stride = pattern[1];
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
        inc = pattern[j] + delta * i;                    
        if((i*delta+(pattern_length - 1) * pattern[1]) >= (iter * 1024)){
                    icn1 = icn1 + (iter * 1024) - (i * delta);
                    iter = iter + 1;
        }        
        if(( inc + icn1 + prev) >= (iter * 1024)){
          flag = 1;
          status = 1;
          prev = icn1 + prev;
          iter = iter + 1;
          break;
        }
      
        if(status){
          status =0;
          prev = prev - ((inc + icn1 + prev) % 1024);
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
          SetRuntimeArgs(program, compute_kernel_handle, core, {n_tiles, pattern_length, delta, extra_tile, stride, single_tile_size, count, wrap});
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
    printf("TT Result : \n");

    for(uint32_t i= 0; i < pattern_length; i++){
        printf("%u ", dev_dense[i]);
    }
    printf("\n\n");
    printf("Expected Result : \n");
    for (size_t i = 0; i < count; ++i){
      for (size_t j = 0; j < pattern_length; ++j){
        dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
        if(i == (count - 1)){
          printf("%u ", (uint32_t)dense[j + pattern_length * (i % wrap)]);
        }
      }
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
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle) {
      
    //Implementation : TBD

    return 1.0;  
}

//Scatter_Gather Kernel
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
    //Implementation : TBD

    return 1.0;
}

//Multi_gather Kernel
template<typename T> 
double metalium_multi_gather_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather, const aligned_vector<double> &sparse, aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{    

    //Implementation : TBD

    return 1.0;
}

//Multi_Scatter Kernel
template<typename T> 
double metalium_multi_scatter_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_scatter, aligned_vector<double> &sparse, const aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle)
{
    
    //Implementation : TBD

    return 1.0;
}
