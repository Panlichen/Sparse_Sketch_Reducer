software requirments
```
pip install numpy scipy sklearn 
```
pycuda
for GTX2080 CUDA10
nvcc --cubin -arch=sm_75 ./codings/LSSkernel.cu