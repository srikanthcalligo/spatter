// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {

    //DeviceZoneScopedN("TEST-FULL");

    uint32_t n_tiles_gather = get_arg_val<uint32_t>(0);
    uint32_t pattern_length =  get_arg_val<uint32_t>(1);    
    uint32_t delta_gather =  get_arg_val<uint32_t>(2);
    uint32_t delta_scatter =  get_arg_val<uint32_t>(3);
    uint32_t extra_tile =  get_arg_val<uint32_t>(4);
    uint32_t stride_gather =  get_arg_val<uint32_t>(5);
    uint32_t stride_scatter =  get_arg_val<uint32_t>(6);
    uint32_t single_tile_size =  get_arg_val<uint32_t>(7);
    uint32_t count = get_arg_val<uint32_t>(8);
    uint32_t wrap = get_arg_val<uint32_t>(9);
    
    uint32_t loop_calc_count = (single_tile_size - ((pattern_length - 1) * stride_gather));
    uint32_t loop_count = 0;
    if(delta_gather){
        loop_count =  loop_calc_count / delta_gather;

        if(loop_calc_count % delta_gather){
            loop_count = loop_count + 1;
        }
    } else {
        loop_count = count;
    }

    constexpr auto cb_sparse_gather = tt::CBIndex::c_0;
    constexpr auto cb_pattern_gather = tt::CBIndex::c_1;
    constexpr auto cb_pattern_scatter = tt::CBIndex::c_2;
    constexpr auto cb_sparse_scatter_inter = tt::CBIndex::c_3;
    constexpr auto cb_sparse_scatter = tt::CBIndex::c_4;

    constexpr uint32_t dst_reg = 0;
    uint32_t scatter_index = 0, gather_index = 0;

    unary_op_init_common(cb_sparse_scatter_inter, cb_sparse_scatter);
    copy_tile_init(cb_sparse_scatter_inter);


    for(uint32_t tile_id = 0; tile_id < n_tiles_gather; tile_id++)
    {
        acquire_dst();

        cb_wait_front(cb_sparse_gather, 1);
        cb_wait_front(cb_pattern_gather, 1);
        cb_wait_front(cb_pattern_scatter, 1);
        cb_wait_front(cb_sparse_scatter_inter, 1);

        volatile uint32_t* pattern_gather_ptr;
        cb_get_tile(cb_pattern_gather, 0, &pattern_gather_ptr);
        pattern_gather_ptr = pattern_gather_ptr + 4; //Need to add 4 because read ptr is off by 1 << 4

        volatile uint32_t* pattern_scatter_ptr;
        cb_get_tile(cb_pattern_scatter, 0, &pattern_scatter_ptr);
        pattern_scatter_ptr = pattern_scatter_ptr + 4;
    
        volatile uint32_t* sparse_scatter_tile_ptr;
        cb_get_tile(cb_sparse_scatter_inter, 0, &sparse_scatter_tile_ptr);
        sparse_scatter_tile_ptr = sparse_scatter_tile_ptr + 4;

        volatile uint32_t* sparse_gather_ptr;
        cb_get_tile(cb_sparse_gather, 0, &sparse_gather_ptr);
        sparse_gather_ptr = sparse_gather_ptr + 4;


        if((tile_id == (n_tiles_gather - 1)) && (extra_tile != 0)){
            loop_count = count - (tile_id * loop_count);
            if((int)loop_count < 0){
                loop_count = 0;
            }
        }

        for (uint32_t i = 0; i < loop_count; i++) {
            scatter_index = delta_scatter * i;
            gather_index = delta_gather * i;
            #pragma GCC unroll 8
            for (uint32_t j = 0; j < pattern_length; j++) {
                // Write from gather to scatter
                //Actual Implementation
                //sparse_scatter_tile_ptr[pattern_scatter_ptr[j] + delta_scatter * i] = sparse_gather_ptr[pattern_gather_ptr[j] + delta_gather * i];
                //Optimized Version 
                sparse_scatter_tile_ptr[j * stride_scatter + scatter_index] = sparse_gather_ptr[ j * stride_gather + gather_index]; 
            }
        }

        copy_tile(cb_sparse_scatter_inter, 0, dst_reg);

        cb_reserve_back(cb_sparse_scatter, 1);

        pack_tile(dst_reg, cb_sparse_scatter);
        
        cb_push_back(cb_sparse_scatter, 1);

        cb_pop_front(cb_sparse_gather, 1);
        cb_pop_front(cb_pattern_gather, 1);
        cb_pop_front(cb_pattern_scatter, 1);
        cb_pop_front(cb_sparse_scatter_inter, 1);

        release_dst();
    }

}
}
