#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0

    //Copy data from DRAM to Core0,0 L1

    uint32_t dram_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_x = get_arg_val<uint32_t>(2);
    uint32_t src0_y =  get_arg_val<uint32_t>(3);
    uint32_t dram_addr1 = get_arg_val<uint32_t>(1);
    uint32_t src1_x = get_arg_val<uint32_t>(4);
    uint32_t src1_y = get_arg_val<uint32_t>(5);
    
    uint32_t n_tiles =  get_arg_val<uint32_t>(6);

    uint32_t dram_addr_out = get_arg_val<uint32_t>(7);
    uint32_t dst_x = get_arg_val<uint32_t>(8);
    uint32_t dst_y =  get_arg_val<uint32_t>(9);

    uint32_t pattern_length =  get_arg_val<uint32_t>(10);
    uint32_t delta =  get_arg_val<uint32_t>(11);
    uint32_t wrap =  get_arg_val<uint32_t>(12);
    uint32_t num_tiles_written = get_arg_val<uint32_t>(13);
    uint32_t num_output_tiles_per_core = get_arg_val<uint32_t>(14);
    uint32_t core_id = get_arg_val<uint32_t>(15);
    
    uint64_t src0_dram_noc_addr = get_noc_addr(src0_x,src0_y,dram_addr);
    uint64_t src1_dram_noc_addr = get_noc_addr(src1_x,src1_y,dram_addr1);
    uint64_t dst_dram_noc_addr = get_noc_addr(dst_x,dst_y,dram_addr_out);
    
    constexpr uint32_t cb_id0 = tt::CB::c_in0;
    constexpr uint32_t cb_id1 = tt::CB::c_in1;
    constexpr uint32_t cb_id2 = tt::CB::c_out0;
    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id1);
    uint32_t ublock_size_bytes_2 = get_tile_size(cb_id2);
    
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id0);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id1);
    uint32_t l1_write_addr_out0 = get_write_ptr(cb_id2);

    const InterleavedAddrGenFast<true> src_a_buf = {
        .bank_base_address = dram_addr,          // The base address of the buffer
        .page_size = ublock_size_bytes_0,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    const InterleavedAddrGenFast<true> dst_a_buf = {
        .bank_base_address = dram_addr_out,          // The base address of the buffer
        .page_size = ublock_size_bytes_2,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    //Read data from DRAM -> L1 circular buffers
    //Read pattern buffer to local thread
    noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
    noc_async_read_barrier();
    uint32_t* dat1 = (uint32_t*) l1_write_addr_in1;
    //Read output buffer to thread
    noc_async_read_tile(core_id, dst_a_buf, l1_write_addr_out0); // read the tile into the circular buffer
    noc_async_read_barrier();
    uint32_t* dat2 = (uint32_t*) l1_write_addr_out0;

    uint32_t outer_loop_count = 1024 / pattern_length; // 32 * 32
  
    //Each thread will iterate through all tiles which have been assigned.
    for(uint32_t tile_id = core_id*num_output_tiles_per_core; tile_id < (core_id*num_output_tiles_per_core+num_output_tiles_per_core); tile_id++) {
        noc_async_read_tile(tile_id, src_a_buf, l1_write_addr_in0); // read the tile into the circular buffer
        noc_async_read_barrier();
        uint32_t* dat0 = (uint32_t*) l1_write_addr_in0;

        for(uint32_t i = 0; i < outer_loop_count ; i++) {
            for(uint32_t j = 0; j < pattern_length; j++) {
                *(dat2 + (j + pattern_length * (i % wrap))) = *(dat0 + (*(dat1+j) + delta * i));
            }
        }       
    }

    // Write data from L1 circulr buffer (out0) -> DRAM
    noc_async_write_tile(core_id, dst_a_buf, l1_write_addr_out0);
    noc_async_write_barrier();
}
