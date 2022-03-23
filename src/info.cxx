#include "omp.h"
#include <cstdio>
#include <iostream>
#include <CL/sycl.hpp>
using namespace sycl;

#include <chrono>


#ifdef NOGPU
static queue Q(cpu_selector{});
#else
static queue Q(gpu_selector{});
#endif


int main(int argc, char* argv[])
{
    std::cout << "  Using SYCL device: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "  Max work group size: " << Q.get_device().get_info<info::device::max_work_group_size>() << std::endl;
    std::cout << "  Max mem alloc: " << Q.get_device().get_info<info::device::max_mem_alloc_size>() << std::endl;
    std::cout << "  max clock freq: " << Q.get_device().get_info<info::device::max_clock_frequency>() << std::endl;
    std::cout << "  max compute units: " << Q.get_device().get_info<info::device::max_compute_units>() << std::endl;
    std::cout << "  max work item dimensions: " << Q.get_device().get_info<info::device::max_work_item_dimensions>() << std::endl;
    std::cout << "  preferred vector width: " << Q.get_device().get_info<info::device::preferred_vector_width_double>() << std::endl;
    //std::cout << "  preferred vector width: " << Q.get_device().get_info<info::kernel_work_group::preferred_work_group_size_multiple>() << std::endl;

    auto info = Q.get_device().get_info<info::device::max_work_item_sizes>();
    std::cout<< "  WG sizes: ";
    for (int i=0;i<3;i++) std::cout << "  : " <<  info[i];
    std::cout<< "\n";

    std::cout << "  OpenMP sees " << omp_get_num_devices() << " devices\n";

    return 0;
}
