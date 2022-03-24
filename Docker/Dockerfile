from ghcr.io/intel/llvm/ubuntu2004_base

RUN apt install software-properties-common wget aptitude -y

RUN wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt update
RUN apt install cuda-toolkit-11-4 -y
ENV PATH=/usr/local/cuda-11.4/bin:${PATH}

WORKDIR /opt
RUN git clone https://github.com/intel/llvm -b sycl --depth 1 --single-branch
#RUN apt install nvidia-cuda-toolkit -y
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:${LD_LIBRARY_PATH}
RUN CUDA_LIB_PATH=/usr/local/cuda-11.4/lib64/stubs python llvm/buildbot/configure.py --cuda -o build --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.4"
RUN python llvm/buildbot/compile.py -o build
ENV PATH=/opt/build/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/build/lib:${LD_LIBRARY_PATH}
WORKDIR /opt/buildomp
RUN cmake  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=/opt/build  -GNinja ../llvm/openmp 
##RUN cmake -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES="75" -DLIBOMPTARGET_ENABLE_DEBUG=YES -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=/opt/build  -GNinja ../llvm/openmp 
RUN ninja
RUN ninja install
RUN pip install jinja2 numpy
