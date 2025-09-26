#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main()
{
    uint32_t sparse_dram_addr       = get_arg_val<uint32_t>(0);
    uint32_t pattern_dram_addr      = get_arg_val<uint32_t>(1);
    uint32_t pattern_gather_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t compute_pattern_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t n_tiles                = get_arg_val<uint32_t>(4);

    constexpr auto sparse_cb_id0          = tt::CBIndex::c_0;
    constexpr auto pattern_cb_id1         = tt::CBIndex::c_1;
    constexpr auto pattern_gather_cb_id1  = tt::CBIndex::c_2;
    constexpr auto compute_pattern_cb_id1 = tt::CBIndex::c_3;

    uint32_t sparse_tile_size       = get_tile_size(sparse_cb_id0);
    uint32_t pattern_tile_size      = get_tile_size(pattern_cb_id1);
    uint32_t pattern_gather_tile_size = get_tile_size(pattern_gather_cb_id1);
    uint32_t compute_pattern_tile_size = get_tile_size(compute_pattern_cb_id1);

    // Read pattern tile
    uint32_t pattern_l1_write_addr = get_write_ptr(pattern_cb_id1);
    noc_async_read(pattern_dram_addr, pattern_l1_write_addr, pattern_tile_size);
    noc_async_read_barrier();

    // Read pattern gather tile
    uint32_t pattern_gather_l1_write_addr = get_write_ptr(pattern_gather_cb_id1);
    noc_async_read(pattern_gather_dram_addr, pattern_gather_l1_write_addr, pattern_gather_tile_size);
    noc_async_read_barrier();

    cb_reserve_back(pattern_cb_id1, 1);
    cb_reserve_back(pattern_gather_cb_id1, 1);
    cb_reserve_back(compute_pattern_cb_id1, 1);

    cb_push_back(pattern_cb_id1, 1);
    cb_push_back(pattern_gather_cb_id1, 1);
    cb_push_back(compute_pattern_cb_id1, 1);


    // Read sparse tiles into CB
    const InterleavedAddrGenFast<true> sparse_src_buf = {
        .bank_base_address = sparse_dram_addr,
        .page_size = sparse_tile_size,
        .data_format = DataFormat::UInt32
    };

    for (uint32_t tile_id = 0; tile_id < n_tiles; tile_id++) {
        
        cb_reserve_back(sparse_cb_id0, 1);

        uint32_t cb_addr = get_write_ptr(sparse_cb_id0);

        noc_async_read_tile(tile_id, sparse_src_buf, cb_addr); //read tile id 
        noc_async_read_barrier();
        cb_push_back(sparse_cb_id0, 1);

    }

}
