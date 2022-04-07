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

RUN apt-get install libpng-dev -y

WORKDIR /opt/intelstuff
RUN wget https://github.com/intel/compute-runtime/releases/download/22.13.22789/intel-gmmlib_22.0.2_amd64.deb
RUN wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.10778/intel-igc-core_1.0.10778_amd64.deb
RUN wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.10778/intel-igc-opencl_1.0.10778_amd64.deb
RUN wget https://github.com/intel/compute-runtime/releases/download/22.13.22789/intel-opencl-icd-dbgsym_22.13.22789_amd64.ddeb
RUN wget https://github.com/intel/compute-runtime/releases/download/22.13.22789/intel-opencl-icd_22.13.22789_amd64.deb
RUN wget https://github.com/intel/compute-runtime/releases/download/22.13.22789/intel-level-zero-gpu-dbgsym_1.3.22789_amd64.ddeb
RUN wget https://github.com/intel/compute-runtime/releases/download/22.13.22789/intel-level-zero-gpu_1.3.22789_amd64.deb
RUN  dpkg -i *.deb

RUN apt install clinfo -y

WORKDIR /opt/intel/oclcpuexp_2022.13.3.0.16
RUN wget https://github.com/intel/llvm/releases/download/2022-WW13/oclcpuexp-2022.13.3.0.16_rel.tar.gz
RUN tar xzf oclcpuexp-2022.13.3.0.16_rel.tar.gz
RUN mkdir -p /etc/OpenCL/vendors
RUN echo /opt/intel/oclcpuexp_2022.13.3.0.16/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd
WORKDIR /opt/intel
RUN wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.5.0/oneapi-tbb-2021.5.0-lin.tgz
RUN tar -zxvf oneapi-tbb*lin.tgz
RUN ln -s /opt/intel/oneapi-tbb-2021.5.0/lib/intel64/gcc4.8/libtbb.so          /opt/intel/oclcpuexp_2022.13.3.0.16/x64
RUN ln -s /opt/intel/oneapi-tbb-2021.5.0/lib/intel64/gcc4.8/libtbbmalloc.so    /opt/intel/oclcpuexp_2022.13.3.0.16/x64
RUN ln -s /opt/intel/oneapi-tbb-2021.5.0/lib/intel64/gcc4.8/libtbb.so.12       /opt/intel/oclcpuexp_2022.13.3.0.16/x64
RUN ln -s /opt/intel/oneapi-tbb-2021.5.0/lib/intel64/gcc4.8/libtbbmalloc.so.2  /opt/intel/oclcpuexp_2022.13.3.0.16/x64
RUN echo /opt/intel/oclcpuexp_2022.13.3.0.16/x64 >> /etc/ld.so.conf.d/libintelopenclexp.conf
RUN ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf

##RUN apt remove intel-gmmlib intel-igc-core intel-igc-opencl intel-level-zero-gpu intel-ocloc intel-opencl libigc-dev libigc -y
#RUN apt install clinfo -y
#RUN apt install strace -y
##RUN apt install beignet-opencl-icd -y
##RUN pip install intel-opencl-rt==2021.3.0

##RUN apt purge intel-opencl-icd -y
##RUN apt purge intel-gmmlib -y

#RUN apt-get update

#WORKDIR /opt/oneapi
#RUN wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146_offline.sh
#RUN apt -y install ncurses-term
#RUN bash l_BaseKit_p_2022.1.2.146_offline.sh -a -s --eula accept
##RUN  sh ./l_BaseKit_p_2022.1.2.146_offline.sh


##RUN apt install lsb-core cpio -y --fix-missing
##WORKDIR /opt/intelopencl
##RUN wget https://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532/l_opencl_p_18.1.0.015.tgz
##RUN tar xzf l_opencl_p_18.1.0.015.tgz
##WORKDIR /opt/intelopencl/l_opencl_p_18.1.0.015
##RUN sed -i "s|decline|accept|g" silent.cfg
##RUN ./install.sh -s silent.cfg 



##RUN add-apt-repository ppa:intel-opencl/intel-opencl -y
##RUN apt update -y
##RUN apt install intel-opencl-icd -y
##RUN apt install clinfo -y
##RUN apt install pocl-opencl-icd -y
 ##apt install intel-opencl-icd -y
 ##apt install beignet-opencl-icd -y

