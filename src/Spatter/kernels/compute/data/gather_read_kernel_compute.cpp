#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //DeviceZoneScopedN("TEST-FULL");
    //export TT_METAL_DPRINT_CORES=0,0
    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t compute_pattern_dram_addr = get_arg_val<uint32_t>(2);    
    uint32_t n_tiles =  get_arg_val<uint32_t>(3);
    
    constexpr uint32_t sparse_cb_id0 = tt::CBIndex::c_0;
    constexpr uint32_t pattern_cb_id1 = tt::CBIndex::c_1;
    constexpr uint32_t compute_pattern_cb_id1 = tt::CBIndex::c_2;

    uint32_t sparse_tile_size = get_tile_size(sparse_cb_id0);
    uint32_t pattern_tile_size = get_tile_size(pattern_cb_id1);
    uint32_t compulte_pattern_tile_size = get_tile_size(compute_pattern_cb_id1);
    
    uint32_t pattern_l1_write_addr_in1 = get_write_ptr(pattern_cb_id1);
    noc_async_read(pattern_dram_addr, pattern_l1_write_addr_in1, pattern_tile_size);
    noc_async_read_barrier();
    
    const InterleavedAddrGenFast<true> sparse_src_buf = {
        .bank_base_address = sparse_dram_addr,          // The base address of the buffer
        .page_size = sparse_tile_size,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    for(uint32_t tile_id = 0; tile_id < n_tiles; tile_id++) {
        //Reserve the memory for the below tiles in CB
        cb_reserve_back(sparse_cb_id0, 1);
        cb_reserve_back(pattern_cb_id1, 1);
        cb_reserve_back(compute_pattern_cb_id1, 1);
        //Read the sparse array tile by tile
        uint32_t cb_in0_addr = get_write_ptr(sparse_cb_id0);
        noc_async_read_tile(tile_id, sparse_src_buf, cb_in0_addr); // read the tile into SRAM
        noc_async_read_barrier();
        //Write below tiles to CB
        cb_push_back(sparse_cb_id0, 1);
        cb_push_back(pattern_cb_id1, 1);
        cb_push_back(compute_pattern_cb_id1, 1);
    }
}
