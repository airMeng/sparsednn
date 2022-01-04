import textwrap
import numpy as np
from code_fragments import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bs', type=int,
                                help='batch size')
    parser.add_argument('--ic', type=int,
                                help='input channel number')
    parser.add_argument('--oc', type=int, default="1024",
                                help='output channel number')
    parser.add_argument('--ocb', type=int,
                                help='block size in oc dim')
    parser.add_argument('--icb', type=int,
                                help='block size in ic dim')
    parser.add_argument('--act_file', type=str,
                               help='cnpy files of activation')
    parser.add_argument('--weight_file', type=str,
                               help='cnpy files of weight')
    parser.add_argument('--bias_file', type=str,
                                help='cnpy files of bias')
    parser.add_argument('--relu', default=False, action='store_true',
                                help='whether support post-op relu')
    parser.add_argument('--sum', default=False, action='store_true',
                                help='whether support post-op append_sum')
    parser.add_argument('--bsb', type=int, 
                                help='block size in bs dim')
    parser.add_argument('--bias_file', type=str,
                                help='cnpy files of bias')

    args = parser.parse_args()
    return args

def load_xmm(ptr, reg):
    if ptr == -1:
        return ""
    else:
        return "vmovdqu8  {}(%r8,%r11,1), %xmm{};\n".format(ptr, reg)

def load_ymm(ptr, reg, mask="k1"):
    if ptr == -1:
        return ""
    else:
        return "vbroadcasti32x4  {}(%r8,%r11,1), %ymm{}, %{}\n".format(ptr, reg, mask)

def reorder_zmm(reg0, reg1, mask="26", index="25"):
    return " vpermt2d  %zmm{}, %zmm25, %zmm{};\n vpshufb  %zmm26, %zmm{}, %zmm{};\n".format(reg1, reg0, reg0, reg0)

def zero_zmm(reg):
    return "vxorps  %zmm{}, %zmm{}, %zmm{};\n".format(reg, reg, reg)

def zmm_broadcast(ptr, reg):
    return "vpbroadcastd {}(%rcx), %zmm{};".format(ptr, reg)

def micro_load_activation(args, acts, reg0, reg1):
    # load 4 1x16 int8, concat and shuffle into one ZMM
    assert len(acts) == 4
    asm = ""
    if -1 in acts:
        asm += zero_zmm(reg)
        asm += zero_zmm("29")
    asm += load_xmm(acts[0], reg0)
    asm += load_ymm(acts[1], reg0)
    asm += load_xmm(acts[2], reg1)
    asm += load_ymm(acts[3], reg1)
    asm += reorder_zmm(reg0, reg1)
    return asm

def micro_load_weight(args, wgt, reg=28):
    # load 4 int8, broadcast into one ZMM 
    assert len(wgts) == 4
    asm = ""
    asm += zmm_broadcast(wgt, reg)
    return asm    
 
def load_activation(args, acts):
    assert len(acts) = args.bsb
    asm = ""
    act_regs = [i for i in range(22, 22+args.bsb)]
    for i, act in enumerate(acts):
        asm += micro_load_activation(args, act, act_regs[i])
    return asm

def load_weight(args, wgts):
    assert len(wgts) = args.ocb
    asm = ""
    wgt_regs = [i for i in range(31, 31 - args.ocb, -1)]
    for i, wgt in enumerate(wgts):
        asm += micro_load_weight(args, wgt, wgt_regs[i])
    return wgts

def micro_kernel(args, act_regs, wgt_regs):
    asm = ""
    out_reg = -1
    for act_reg in act_regs:
        for wgt_reg in wgt_regs:
            asm += "vpdpbusd %zmm{},%zmm{},%zmm{};".format(act_reg, wgt_reg, out_reg+=1)
    return asm

def generate_asm(args, oc_idxs, ic_ptrs):
    asm = """
    ..B1.NUM1:
    xorl    %r10d, %r10d;
    ..B1.NUM2:
    imul      $16,  %r10d, %r11d;
    add       %r9d, %r11d;
    movslq  %r11d, %r11;
    add     $CT,    %r10d;

    """.replace("NUM1",str(BB_offset + block*2+2)).replace("NUM2",str(BB_offset + block * 2 + 3)).replace("STRIDE",str(8)).replace("CT",str(CT))

    for oc_idx, ic_ptr in zip(oc_idxs, ic_ptrs):
        assert len(oc_idx) == len(ic_ptr)
        num_idx = len(oc_idx)
        num_padding = 4 - num_idx % 4 if num_idx % 4 != 0 else 0
        oc_idx_padded = oc_idx
        oc_idx_padded.extend([-1] * num_padding)
        ic_ptr_padded = ic_ptr
        ic_ptr_padded.extend([-1] * num_padding)
        for i in range(0, len(ic_ptr_padded), 4):
            ic_indices = ic_ptr[i:(i+4)
            
        








def main(args):
   weight = np.load(args.weight_file)
   activation = np.load(args.act_file)
   bias = np.load(args.bias_file)
   assert(weight.shape[0]==args.oc) 
   assert(weight.shape[1]==activation.shape[0])
   assert(activation.shape[1]==args.bs)
   assert(weight.shape[1]==args.ic)
   bsb_n = args.bs // args.bs % args.bsb == 0 else args.bs // args.bsb + 1
   bs = args.bs
   ic = args.ic
   oc = args.oc
   asm_program = """
# -- Begin  _spmm
        .text
# mark_begin;
       .align    16,0x90
        .globl _spmm
# --- mm(void *)
_spmm:
# parameter 1: %rdi
..B1.1:                         # Preds ..B1.0
                                # Execution count [9.00e-01]
        .cfi_startproc
..___tag_value__spmm.1:
..L2:
                                                          #45.1
        pushq     %rbp                                          #45.1
        .cfi_def_cfa_offset 16
        movq      %rsp, %rbp                                    #45.1
        .cfi_def_cfa 6, 16
        .cfi_offset 6, -16
        andq      $-32, %rsp                                    #45.1
        subq      $96, %rsp                                     #45.1
        mov         $0xf0 , %ebx;
        kmovb       %ebx, %k1
        movq      (%rdi), %rcx                                  # the first argument which is packed nonzero values pointer
        movq      8(%rdi), %rsi                                 # the second argument which is bias values pointer
        movq      16(%rdi), %r8                                 # the third argument which is input matrix pointer
        movq      24(%rdi), %rdx                                # the fourth argument which is output matrix pointer
        movq      32(%rdi), %rbx                                # the scale
        movl      44(%rdi), %eax                                # end iteration count in the C dimension, useful for multithreading
        movl      40(%rdi), %edi                                # start iteration count in the C dimension, useful for multithreading
        decl    %eax
        decl    %edi
        imul     $BSB_N, %eax, %r9d

        vpmovzxbd   vpermt2d_control(%rip), % zmm25;            # initialize the control avx vectors which we are going to use for permutes and shuffles
        vbroadcasti32x4   vpshufb_control(%rip), % zmm26;

    """.replace("BSB_N",str(bsb_n))
    bounds = [args.ocb * i for i in range(oc)] + [args.oc]  # bounds of each oc block
    ocb = args.ocb
    icb_n = ic // args.icb
    ocb_n = oc // args.ocb
    weight_reshape = weight.transpose().reshape(icb_n, icb, ocb_n, ocb).transpose(0, 2, 1, 3).reshape(icb_n, ocb_n, -1).sum(-1)
    weight_per_col = [weight_reshape[:, i] for i in range(ic)]
    # weight_nonzero_blocks = [int(np.count_nonzero(weight_reshape[i, :])) for i in range(ic)]
    # BSR here for weight
    oc_idx = [weight_per_col_i.nonzero()[1] for weight_per_col_i in weight_per_col]
    ic_ptr = [range(icb * ocb) for _ in range(np.concatenate(oc_idx).size))]
    oc_idx = np.repeat(oc_idx, icb * ocb)
    # oc_ptr = np.concatenate((np.array([0]), np.cumsum(weight_nonzero_blocks)))
    # for icb_i in icb_n:
    #     ic_offset = icb_i * ocb_n * 2
    #     for ocb_i in ocb_n:
    #         oc_offset = bounds[ocb_i]
    #         oc_block = bounds[ocb_i + 1] - oc_offset

    #         oc_indices,
    asm_program += generate_asm(args, oc_idx, ic_ptr)
