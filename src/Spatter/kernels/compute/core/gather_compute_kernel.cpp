// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_in1 = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_3;
    constexpr uint32_t dst_reg = 0;

    unary_op_init_common(cb_in1, cb_out0);
    copy_tile_init(cb_in1);
    acquire_dst();
    for(uint32_t i = 0; i < n_tiles; i++)
    {
        cb_wait_front(cb_in1, 1);
        copy_tile(cb_in1, 0, dst_reg);
        cb_pop_front(cb_in1, 1);   
    }
    
    cb_reserve_back(cb_out0, 1);

    pack_tile(dst_reg, cb_out0);
    
    cb_push_back(cb_out0, 1);
    
    release_dst();
}
}