#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    //export TT_METAL_DPRINT_CORES=0,0
    uint32_t sparse_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t pattern_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t dense_dram_addr = get_arg_val<uint32_t>(2);    
    uint32_t n_tiles =  get_arg_val<uint32_t>(3);
    uint32_t pattern_length =  get_arg_val<uint32_t>(4);    
    uint32_t delta =  get_arg_val<uint32_t>(5);
    uint32_t extra_tile =  get_arg_val<uint32_t>(6);
    uint32_t stride =  get_arg_val<uint32_t>(7);
    uint32_t single_tile_size =  get_arg_val<uint32_t>(8);
    uint32_t count = get_arg_val<uint32_t>(9);
    uint32_t wrap = get_arg_val<uint32_t>(10);

    constexpr uint32_t sparse_cb_id0 = tt::CBIndex::c_0;
    constexpr uint32_t pattern_cb_id1 = tt::CBIndex::c_1;
    constexpr uint32_t dense_cb_id1 = tt::CBIndex::c_2;

    uint32_t sparse_tile_size = get_tile_size(sparse_cb_id0);
    uint32_t pattern_tile_size = get_tile_size(pattern_cb_id1);
    uint32_t dense_tile_size = get_tile_size(dense_cb_id1);
    
    uint32_t pattern_l1_write_addr_in1 = get_write_ptr(pattern_cb_id1);
    noc_async_read(pattern_dram_addr, pattern_l1_write_addr_in1, pattern_tile_size);
    noc_async_read_barrier();
    uint32_t dense_l1_write_addr_in1 = get_write_ptr(dense_cb_id1);
    
    const InterleavedAddrGenFast<true> sparse_src_buf = {
        .bank_base_address = sparse_dram_addr,          // The base address of the buffer
        .page_size = sparse_tile_size,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    const InterleavedAddrGenFast<true> dense_buf = {
        .bank_base_address = dense_dram_addr,          // The base address of the buffer
        .page_size = dense_tile_size,         // The size of a buffer page
        .data_format = DataFormat::UInt32, // The data format of the buffer
    };

    uint32_t* data_pattern = (uint32_t*) pattern_l1_write_addr_in1;
    uint32_t* compute_dense = (uint32_t*) dense_l1_write_addr_in1;
    uint32_t loop_count = single_tile_size / delta;
    uint32_t extra_itr = 0;

    if(pattern_length % delta){
        extra_itr = 1;
    }

    loop_count = loop_count - extra_itr - (stride - 1);

    for(uint32_t tile_id = 0; tile_id < n_tiles; tile_id++) {
        uint32_t cb_in0_addr = get_write_ptr(sparse_cb_id0);
        noc_async_read_tile(tile_id, sparse_src_buf, cb_in0_addr); // read the tile into the circular buffer
        noc_async_read_barrier();
        uint32_t* data = (uint32_t*)cb_in0_addr;

        if((tile_id == (n_tiles - 1)) && (extra_tile != 0)){
            loop_count = count - (tile_id * loop_count);
        }
        for(uint32_t i = 0; i < loop_count; i++){
            for(uint32_t j = 0; j < pattern_length; j++){
                *(compute_dense + (j + pattern_length * (i % wrap))) = *(data + (*(data_pattern+j) + delta * i));
            }
        }
    }
    //Write last tile data to the DRAM
    noc_async_write_tile(0, dense_buf, dense_l1_write_addr_in1);
    noc_async_write_barrier();
}