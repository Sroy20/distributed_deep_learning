# distributed_deep_learning
In this repository, I will combine PyTorch and Horovod to create high-performance distributed deep learning algorithms. Some of the vanilla codes have been taken from here (https://github.com/kuangliu/pytorch-cifar) which I converted to having distributed learning capability for training on multiple GPUs.

I have used the Horovod (https://github.com/uber/horovod) package released by Uber for enabling distributed deep learning. 

For running the codes use:
mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python run_cifar10.py 


