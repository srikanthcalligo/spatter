#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0
    uint32_t dense_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t scatter_pattern_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t n_tiles =  get_arg_val<uint32_t>(4);
    uint32_t num_tiles_written =  get_arg_val<uint32_t>(5);
    uint32_t num_output_tiles_per_core =  get_arg_val<uint32_t>(6);

    constexpr uint32_t dense_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t pattern_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t scatter_pattern_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t sparse_inter_cb_id = tt::CBIndex::c_3;

    uint32_t sparse_tile_size = get_tile_size(sparse_inter_cb_id);
    uint32_t pattern_tile_size = get_tile_size(pattern_cb_id);
    uint32_t dense_tile_size = get_tile_size(dense_cb_id);
    uint32_t scatter_pattern_tile_size = get_tile_size(scatter_pattern_cb_id);
    
    uint32_t pattern_l1_write_addr = get_write_ptr(pattern_cb_id);
    noc_async_read(pattern_dram_addr, pattern_l1_write_addr, pattern_tile_size);
    noc_async_read_barrier();

    uint32_t scatter_pattern_l1_write_addr = get_write_ptr(scatter_pattern_cb_id);
    noc_async_read(scatter_pattern_dram_addr, scatter_pattern_l1_write_addr, scatter_pattern_tile_size);
    noc_async_read_barrier();
    
    uint32_t dense_l1_write_addr = get_write_ptr(dense_cb_id);
    noc_async_read(dense_dram_addr, dense_l1_write_addr, dense_tile_size);
    noc_async_read_barrier();
    
    for(uint32_t tile_id = num_tiles_written; tile_id < (num_tiles_written+num_output_tiles_per_core); tile_id++) {
        cb_reserve_back(sparse_inter_cb_id, 1);
        cb_reserve_back(dense_cb_id, 1);
        cb_reserve_back(pattern_cb_id, 1);
        cb_reserve_back(scatter_pattern_cb_id, 1);

        cb_push_back(sparse_inter_cb_id, 1);
        cb_push_back(dense_cb_id, 1);
        cb_push_back(pattern_cb_id, 1);
        cb_push_back(scatter_pattern_cb_id, 1);        
    }
   
}