# grayscalepng

A simple SYCL demonstrator.

Read in an RGB-channel PNG and write grayscale image with weightin 0.3 R, 0.59 G 0.11 B.

## Dependencies
Requires CImg.h (https://cimg.eu) and libpng.

##  Build
```
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda--sm_75 -lpng grayscalepng.cxx -o grayscalepng_sm75
```

## Usage

```
./grayscalepng_sm75 lenna.png gray.png
```
