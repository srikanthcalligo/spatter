#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0
    uint32_t pgather_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t dense_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t n_tiles =  get_arg_val<uint32_t>(4);
    uint32_t pattern_length =  get_arg_val<uint32_t>(5);
    uint32_t delta =  get_arg_val<uint32_t>(6);
    uint32_t wrap =  get_arg_val<uint32_t>(7);
    uint32_t single_tile_size = get_arg_val<uint32_t>(8);
    uint32_t extra_tile = get_arg_val<uint32_t>(9);
    uint32_t stride = get_arg_val<uint32_t>(10);
    uint32_t num_tiles_written = get_arg_val<uint32_t>(11);
    uint32_t num_output_tiles_per_core = get_arg_val<uint32_t>(12);
    uint32_t core_id = get_arg_val<uint32_t>(13);
    uint32_t count = get_arg_val<uint32_t>(14);

    //DPRINT << "Core id = " << core_id << " " << num_output_tiles_per_core << " " << num_tiles_written << ENDL();

    constexpr uint32_t cb_id0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_id3 = tt::CBIndex::c_3;
    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id1);
    uint32_t ublock_size_bytes_2 = get_tile_size(cb_id2);
    uint32_t ublock_size_bytes_3 = get_tile_size(cb_id3);
    
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id0);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id1);
    uint32_t l1_write_addr_in2 = get_write_ptr(cb_id2);
    uint32_t l1_write_addr_out0 = get_write_ptr(cb_id3);

    const InterleavedAddrGenFast<true> sparse_buf = {
        .bank_base_address = sparse_dram_addr,          // The base address of the buffer
        .page_size = ublock_size_bytes_2,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    const InterleavedAddrGenFast<true> dense_buf = {
        .bank_base_address = dense_dram_addr,          // The base address of the buffer
        .page_size = ublock_size_bytes_3,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };
    // Read data from DRAM -> L1 circular buffers
    noc_async_read(pattern_dram_addr, l1_write_addr_in0, ublock_size_bytes_0);
    noc_async_read_barrier();
    noc_async_read(pgather_dram_addr, l1_write_addr_in1, ublock_size_bytes_1);
    noc_async_read_barrier();
    //noc_async_read(dense_dram_addr, l1_write_addr_out0, ublock_size_bytes_3);
    noc_async_read_tile(core_id, dense_buf, l1_write_addr_out0); // read the tile into the circular buffer
    noc_async_read_barrier();

    uint32_t* pattern_data = (uint32_t*) l1_write_addr_in0;
    uint32_t* pgather_data = (uint32_t*) l1_write_addr_in1;
    uint32_t* dense_data = (uint32_t*) l1_write_addr_out0;

    uint32_t outer_loop_count = single_tile_size / delta;

    uint32_t extra_tile_inc = extra_tile / delta;
    extra_tile_inc = (extra_tile % delta) ? extra_tile_inc + 1 : extra_tile_inc;

    if((extra_tile != 0) && (stride > 1)){
        outer_loop_count = outer_loop_count - stride + 1;
        extra_tile_inc = 0;
    }
    
    for(uint32_t tile_id = num_tiles_written; tile_id < (num_tiles_written+num_output_tiles_per_core); tile_id++) {
        noc_async_read_tile(tile_id, sparse_buf, l1_write_addr_in2); // read the tile into the circular buffer
        noc_async_read_barrier();
        uint32_t* sparse_data = (uint32_t*) l1_write_addr_in2;
        if((extra_tile != 0) && (tile_id == n_tiles - 1)){
            outer_loop_count = ((n_tiles - 1 ) * outer_loop_count == count ) ? 1 : extra_tile_inc + 1;
        }
        for(uint32_t i = 0; i < outer_loop_count; i++) {
            for(uint32_t j = 0; j < pattern_length; j++) {
                uint32_t index_0 = j + pattern_length * (i % wrap);
                uint32_t index_1 = *(pattern_data + (*(pgather_data + j))) + delta * i;
                *(dense_data + index_0) = *(sparse_data + index_1);
            }
        }
    }
    // Write data from L1 circulr buffer (out0) -> DRAM
    noc_async_write_tile(core_id, dense_buf, l1_write_addr_out0);
    noc_async_write_barrier();
}