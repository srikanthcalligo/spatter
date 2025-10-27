#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0

    //Copy data from DRAM to Core0,0 L1
    //DeviceZoneScopedN("TEST-FULL");

    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t pattern_gather_dram_addr = get_arg_val<uint32_t>(2);    
    uint32_t compute_pattern_dram_addr = get_arg_val<uint32_t>(3);        
    uint32_t num_tiles_written = get_arg_val<uint32_t>(4);
    uint32_t num_output_tiles_per_core = get_arg_val<uint32_t>(5);


    constexpr uint32_t sparse_cb_id0 = tt::CBIndex::c_0;
    constexpr uint32_t pattern_cb_id1 = tt::CBIndex::c_1;
    constexpr uint32_t pattern_gather_cb_id1 = tt::CBIndex::c_2;
    constexpr uint32_t compute_pattern_cb_id1 = tt::CBIndex::c_3;


    uint32_t sparse_tile_size = get_tile_size(sparse_cb_id0);
    
    uint32_t pattern_tile_size = get_tile_size(pattern_cb_id1);
    uint32_t pattern_gather_tile_size = get_tile_size(pattern_gather_cb_id1);
    uint32_t compulte_pattern_tile_size = get_tile_size(compute_pattern_cb_id1);

    uint32_t pattern_l1_write_addr_in1 = get_write_ptr(pattern_cb_id1);
    noc_async_read(pattern_dram_addr, pattern_l1_write_addr_in1, pattern_tile_size);
    noc_async_read_barrier();

    uint32_t pattern_gather_l1_write_addr_in1 = get_write_ptr(pattern_gather_cb_id1);
    noc_async_read(pattern_gather_dram_addr, pattern_gather_l1_write_addr_in1, pattern_gather_tile_size);
    noc_async_read_barrier();
    
    const InterleavedAddrGenFast<true> sparse_src_buf = {
        .bank_base_address = sparse_dram_addr,          // The base address of the buffer
        .page_size = sparse_tile_size,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    for(uint32_t tile_id = num_tiles_written; tile_id < (num_tiles_written+num_output_tiles_per_core); tile_id++) {
        cb_reserve_back(sparse_cb_id0, 1);
        cb_reserve_back(pattern_cb_id1, 1);
        cb_reserve_back(pattern_gather_cb_id1, 1);
        cb_reserve_back(compute_pattern_cb_id1, 1);

        uint32_t cb_in0_addr = get_write_ptr(sparse_cb_id0);
        noc_async_read_tile(tile_id, sparse_src_buf, cb_in0_addr); // read the tile into SRAM
        noc_async_read_barrier();
        cb_push_back(sparse_cb_id0, 1);
        cb_push_back(pattern_cb_id1, 1);
        cb_push_back(pattern_gather_cb_id1, 1);
        cb_push_back(compute_pattern_cb_id1, 1);
    }
}
