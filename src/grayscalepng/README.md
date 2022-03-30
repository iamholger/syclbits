# grayscalepng

A simple SYCL demonstrator.

Read in an RGB-channel PNG and write grayscale image with weightin 0.3 R, 0.59 G 0.11 B.

Input             |  Output
:-------------------------:|:-------------------------:
![lenna](https://user-images.githubusercontent.com/10545425/160864858-caa9a1d6-a109-4382-adbc-63b7ebd91a3f.png) | ![gray](https://user-images.githubusercontent.com/10545425/160864970-d017f92b-2b5f-4836-9542-37d793e009a0.png)


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
