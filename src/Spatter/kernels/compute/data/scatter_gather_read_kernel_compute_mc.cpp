#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0
    uint32_t sparse_gather_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_gather_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t pattern_scatter_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t sparse_scatter_inter_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t n_tiles_gather =  get_arg_val<uint32_t>(4);
    uint32_t num_tiles_written =  get_arg_val<uint32_t>(5);
    uint32_t num_output_tiles_per_core =  get_arg_val<uint32_t>(6);

    constexpr uint32_t sparse_gather_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t pattern_gather_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t pattern_scatter_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t sparse_scatter_inter_cb_id = tt::CBIndex::c_3;

    uint32_t sparse_gather_tile_size = get_tile_size(sparse_gather_cb_id);
    uint32_t pattern_gather_tile_size = get_tile_size(pattern_gather_cb_id);
    uint32_t pattern_scatter_tile_size = get_tile_size(pattern_scatter_cb_id);
    
    // Allocate pattern tiles in CB and read from DRAM
    uint32_t pattern_gather_l1_ptr = get_write_ptr(pattern_gather_cb_id);
    noc_async_read(pattern_gather_dram_addr, pattern_gather_l1_ptr, pattern_gather_tile_size);
    noc_async_read_barrier();

    uint32_t pattern_scatter_l1_ptr = get_write_ptr(pattern_scatter_cb_id);
    noc_async_read(pattern_scatter_dram_addr, pattern_scatter_l1_ptr, pattern_scatter_tile_size);
    noc_async_read_barrier();

    // Prepare sparse gather buffer address generator
    const InterleavedAddrGenFast<true> sparse_src_buf = {
        .bank_base_address = sparse_gather_dram_addr, // DRAM base
        .page_size         = sparse_gather_tile_size,
        .data_format       = DataFormat::UInt32
    };
    
    for(uint32_t tile_id = num_tiles_written; tile_id < (num_tiles_written+num_output_tiles_per_core); tile_id++) {
        cb_reserve_back(sparse_gather_cb_id, 1);
        cb_reserve_back(pattern_gather_cb_id, 1);
        cb_reserve_back(pattern_scatter_cb_id, 1);
        cb_reserve_back(sparse_scatter_inter_cb_id, 1);

        uint32_t cb_in0_addr = get_write_ptr(sparse_gather_cb_id);
        noc_async_read_tile(tile_id, sparse_src_buf, cb_in0_addr); // read the tile into SRAM
        noc_async_read_barrier();
    
        cb_push_back(sparse_gather_cb_id, 1);
        cb_push_back(pattern_gather_cb_id, 1);
        cb_push_back(pattern_scatter_cb_id, 1);
        cb_push_back(sparse_scatter_inter_cb_id, 1);
    }
   
}