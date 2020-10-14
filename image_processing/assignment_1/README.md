```
conda install tqdm
conda install -c conda-forge scikit-image
```
PyTorch with CUDA:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
If we want to use the CPU only (not CUDA) we can run this command instead of the above: `$ conda install pytorch torchvision cpuonly -c pytorch`
