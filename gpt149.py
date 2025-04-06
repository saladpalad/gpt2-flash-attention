import argparse
import time
import math
import random
import inspect
from dataclasses import dataclass
import sys, getopt
import os
from os import getcwd, path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere


NUM_THREADS=8
torch.set_num_threads(NUM_THREADS)

print("\nCompiling code into a PyTorch module...\n\n")
flash_attention = load(name="custom_module", sources=["attention.cu"],  extra_cflags=["-O3"])
correctness_error_message = "\n-------------------------------------------\n YOUR ATTENTION PRODUCED INCORRECT RESULTS"

class CustomAttention(nn.Module):
    def __init__(self, Q,K,V, B, H, N, d, isRef=False, bc=32, br=32):
        super(nn.Module, self).__init__()
        self.Q=Q.cuda()
        self.K=K.cuda()
        self.V=V.cuda()
        self.bc=bc
        self.br=br
        self.B=B
        self.H=H
        self.N=N
        self.d=d
        self.isRef=isRef

    def myFlashAttention(self):
        B, H, N, d = self.B, self.H, self.N, self.d
#        B, H, N, d = 1, 4, 2, 32

        O = torch.zeros((B, H, N, d), device='cuda')
        l = torch.zeros((B, H, N), device='cuda')
        m = torch.full((B, H, N), -float('inf'), device='cuda')

        if self.isRef:
            with record_function("STUDENT - FLASH ATTENTION"):
                out = flash_attention.forward(self.Q, self.K, self.V, O, l, m, B, H, N, d)
            return out
        with record_function("REFERENCE - FLASH ATTENTION"):
            out = flash_attention.forward(self.Q, self.K, self.V, O, l, m, B, H, N, d)
        return out

def createQKVSimple(N,d,B,H):
    Q = torch.empty(B,H,N,d)
    K = torch.empty(B,H,d,N)
    V = torch.empty(B,H,N,d)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for j in range(d):
                    Q[b][h][i][j] = 0.0002 * i + 0.0001 * j
                    K[b][h][j][i] = 0.0006 * i + 0.0003 * j
                    V[b][h][i][j] = 0.00015 * i + 0.0008 * j
    K=K.transpose(-2,-1)
    return Q,K,V

def naive_attn(q, k, v):
    #att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = (q @ k.transpose(-2, -1))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def testTemplate(customFunc, params, test_key):
    N, d, B, H = params

    # Create tensors and move to GPU
    Q, K, V = createQKVSimple(N, d, B, H)
    Q, K, V = Q.cuda(), K.cuda(), V.cuda()
    
    # Compute reference result
    torch.cuda.synchronize()
    start = time.time()
    QKV = naive_attn(Q, K, V)
    torch.cuda.synchronize()
    end = time.time()
    manual_time = end - start
    
    with profile(activities=[
            ProfilerActivity.CUDA,
            ProfilerActivity.CPU
        ],
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            torch.cuda.synchronize()
            start = time.time()
            
            QKS1 = customFunc()
            
            torch.cuda.synchronize()
            end = time.time()
            flash_time = end - start
    
    assert torch.allclose(QKV, QKS1, rtol=0, atol=1e-02), correctness_error_message
    #print("Manual Attention Execution Time: ", f"{manual_time*1000:.4f} ms\n")
    #print("Flash Attention Execution Time: ", f"{flash_time*1000:.4f} ms\n")
    
    # Print profiler results
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
    
    r = prof.key_averages()
    for rr in r:
        if rr.key == test_key:
            key = rr.key
            # Use device_time instead of cuda_time
            device_time = rr.cpu_time + getattr(rr, 'device_time', 0)
            # Use device_memory_usage instead of cuda_memory_usage
            device_mem = getattr(rr, 'device_memory_usage', 0)
            print(f"{test_key} statistics")
            print("device time: ", f"{device_time / 1000.0:.4f}ms")
            print("device mem usage: ", device_mem, "bytes")
    
    # Optional: clean up GPU memory
    torch.cuda.empty_cache()

def test(N, d, B, H, bc, br):
    print("Running Test: Flash Attention\n")
    Q,K,V = createQKVSimple(N,d,B,H)
    attentionModuleStudent = CustomAttention(Q,K,V, B, H, N, d, False, bc, br)
    attentionModuleReference = CustomAttention(Q,K,V, B, H, N, d, True, bc, br)
    params = (N, d, B, H)
    print("-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    testTemplate(attentionModuleStudent.myFlashAttention, params, "REFERENCE - FLASH ATTENTION")
    time.sleep(3)
    print("-----RUNNING STUDENT IMPLEMENTATION-----\n")
    testTemplate(attentionModuleReference.myFlashAttention, params, "STUDENT - FLASH ATTENTION")

def main():

    #d=64 # token embedding length
    #B=16
    #H=12
    B=1
    H=4
    d=32

    parser = argparse.ArgumentParser()
    parser.add_argument("testname", default="part4")
    #parser.add_argument("testname", default="part4")
    parser.add_argument("-m", "--model", default="shakes128", help="name of model to use: shakes128, shakes1024, shakes2048, kayvon")
    parser.add_argument("--inference", action="store_true", default=False, help="run gpt inference")
    #parser.add_argument("-bc",  default="256", help="Flash Attention Bc Size")
    #parser.add_argument("-br", default="256", help="Flash Attention Br Size")
    parser.add_argument("-N", default="1024", help="Flash Attention Sequence Length")

    args = parser.parse_args()

    if args.model == "shakes128":
        N = 128
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes256":
        N = 256
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes1024":
        N = 1024
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes2048":
        N = 2048
        model_filename = "out-shakespeare-char2048Good"
    else:
        print("Unknown model name: %s" % args.model)
        return
    
    if args.inference == False:
        N = 2
        bc, br = 32,32
        test(N, d, B, H, bc, br)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)

if __name__ == "__main__":
    main()
