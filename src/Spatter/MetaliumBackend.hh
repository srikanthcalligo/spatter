#ifndef METALIUM_BACKEND_HH
#define METALIUM_BACKEND_HH

#include <cstddef>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

//Metalium API related functions
CBHandle MakeCircularBuffer_UInt32(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, tt::CBIndex cb_index, uint32_t num_tiles_per_cb, uint32_t single_tile_size);
CBHandle MakeCircularBuffer_BFloat16(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, tt::CBIndex cb_index, uint32_t num_tiles_per_cb, uint32_t single_tile_size);
CBHandle MakeCircularBuffer_Float32(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, tt::CBIndex cb_index, uint32_t num_tiles_per_cb, uint32_t single_tile_size);

std::shared_ptr<tt::tt_metal::Buffer> MakeBuffer_UInt32(IDevice *device, uint32_t size, uint32_t page_size);
std::shared_ptr<tt::tt_metal::Buffer> MakeBuffer_BFloat16(IDevice *device, uint32_t size, uint32_t page_size);
KernelHandle Make_Read_NOC0_Kernel(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, std::string kernel_file);
KernelHandle Make_Write_NOC1_Kernel(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, std::string kernel_file);
KernelHandle Make_Compute_Core_Kernel(CoreCoord core, CoreRangeSet core_set, uint32_t is_parallel_mode_on, Program &program, std::string kernel_file, bool f32_dest_acc_en = false, bool math_approx_mode = false);


//Spatter functions
template<typename T>
double metalium_gather_wrapper(const aligned_vector<size_t> &pattern, const aligned_vector<double> &sparse,
    aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count,bool is_compute_mode_on, uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device, CommandQueue& cq, Program &program, uint32_t single_tile_size,
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template<typename T> 
double metalium_scatter_wrapper(const aligned_vector<size_t> &pattern, aligned_vector<double> &sparse,
    const aligned_vector<double> &dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template<typename T> 
double metalium_scatter_gather_wrapper(const aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse_scatter, const aligned_vector<size_t> &pattern_gather,
    const aligned_vector<double> &sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template<typename T> 
double metalium_multi_gather_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather, const aligned_vector<double> &sparse, aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);

template<typename T> 
double metalium_multi_scatter_wrapper(const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_scatter, aligned_vector<double> &sparse, const aligned_vector<double> &dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, bool is_compute_mode_on,uint32_t is_parallel_mode_on,
    size_t step_size, size_t is_nr_enabled,
    CoreCoord core, uint32_t device_id, IDevice *device,
    CommandQueue& cq, Program &program, uint32_t single_tile_size, 
    KernelHandle data_read_kernel_handle, KernelHandle data_write_kernel_handle, KernelHandle compute_kernel_handle);
#endif
