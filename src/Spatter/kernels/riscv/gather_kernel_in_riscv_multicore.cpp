#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES="(0,0)-(7,7)"
    //Copy data from DRAM to Core0,0 L1
    uint32_t dram_addr_sparse = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_pattern = get_arg_val<uint32_t>(1);
    uint32_t n_tiles =  get_arg_val<uint32_t>(2);
    uint32_t dram_addr_dense = get_arg_val<uint32_t>(3);
    uint32_t pattern_length =  get_arg_val<uint32_t>(4);
    uint32_t delta =  get_arg_val<uint32_t>(5);
    uint32_t wrap =  get_arg_val<uint32_t>(6);
    uint32_t num_tiles_written = get_arg_val<uint32_t>(7);
    uint32_t num_output_tiles_per_core = get_arg_val<uint32_t>(8);
    uint32_t core_id = get_arg_val<uint32_t>(9);
    uint32_t single_tile_size = get_arg_val<uint32_t>(10);
    uint32_t extra_tile = get_arg_val<uint32_t>(11);
    uint32_t stride = get_arg_val<uint32_t>(12);
    
    constexpr uint32_t cb_id0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id2 = tt::CBIndex::c_2;
    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id1);
    uint32_t ublock_size_bytes_2 = get_tile_size(cb_id2);
    
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id0);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id1);
    uint32_t l1_write_addr_out0 = get_write_ptr(cb_id2);

    const InterleavedAddrGenFast<true> src_a_buf = {
        .bank_base_address = dram_addr_sparse,          // The base address of the buffer
        .page_size = ublock_size_bytes_0,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    const InterleavedAddrGenFast<true> src_pat_buf = {
        .bank_base_address = dram_addr_pattern,          // The base address of the buffer
        .page_size = ublock_size_bytes_1,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    const InterleavedAddrGenFast<true> dst_a_buf = {
        .bank_base_address = dram_addr_dense,          // The base address of the buffer
        .page_size = ublock_size_bytes_2,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    //Read data from DRAM -> L1 circular buffers
    //Read pattern buffer to local thread
    noc_async_read_tile(0, src_pat_buf, l1_write_addr_in1);
    noc_async_read_barrier();
    uint32_t* data_pattern = (uint32_t*) l1_write_addr_in1;
    //Read output buffer to thread
    noc_async_read_tile(core_id, dst_a_buf, l1_write_addr_out0); // read the tile into the circular buffer
    noc_async_read_barrier();
    uint32_t* data_dense = (uint32_t*) l1_write_addr_out0;

    uint32_t outer_loop_count = single_tile_size / delta; // 32 * 32
  
    if(extra_tile){
        outer_loop_count = outer_loop_count - stride;
    }
    
    //Each thread will iterate through all tiles which have been assigned to it.
    for(uint32_t tile_id = num_tiles_written; tile_id < (num_tiles_written+num_output_tiles_per_core); tile_id++) {
        noc_async_read_tile(tile_id, src_a_buf, l1_write_addr_in0); // read the tile into the circular buffer
        noc_async_read_barrier();
        uint32_t* data_sparse = (uint32_t*) l1_write_addr_in0;

        if((extra_tile != 0) && (tile_id == (n_tiles -1))){
            outer_loop_count = 1;
        }

        for(uint32_t i = 0; i < outer_loop_count ; i++) {
            for(uint32_t j = 0; j < pattern_length; j++) {
                *(data_dense + (j + pattern_length * (i % wrap))) = *(data_sparse + (*(data_pattern+j) + delta * i));
            }
        }       
    }
    // Write data from L1 circulr buffer (out0) -> DRAM
    noc_async_write_tile(core_id, dst_a_buf, l1_write_addr_out0);
    noc_async_write_barrier();
}
