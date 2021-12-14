#include "omp.h"
#include <cstdio>
#include <iostream>
#include <CL/sycl.hpp>
using namespace sycl;

void flux(
 const double * __restrict__ Q, // Q[5+0],
 int                                          normal,
 double * __restrict__ F // F[5],
) {
  constexpr double gamma = 1.4;
  const double irho = 1./Q[0];
  #if Dimensions==3
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]+Q[3]*Q[3]));
  #else
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]));
  #endif

  const double coeff = irho*Q[normal+1];
  F[0] = coeff*Q[0];
  F[1] = coeff*Q[1];
  F[2] = coeff*Q[2];
  F[3] = coeff*Q[3];
  F[4] = coeff*Q[4];
  F[normal+1] += p;
  F[4]        += coeff*p;
}


double maxEigenvalue(
  const double * __restrict__ Q,
  int                                          normal
) {
  constexpr double gamma = 1.4;
  const double irho = 1./Q[0];
  #if Dimensions==3
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]+Q[3]*Q[3]));
  #else
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]));
  #endif

  const double u_n = Q[normal + 1] * irho;
  const double c   = std::sqrt(gamma * p * irho);

  double result = std::max( std::abs(u_n - c), std::abs(u_n + c) );
  return result;
}


template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void hcompute(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
  const double dt =0.5;
  const size_t NPT=20;

  Q.submit([&](handler &cgh)
  {
    //sycl::stream out(100000, 768, cgh);
    //cgh.parallel_for_work_group(range<3>{NPT, numVPAIP, numVPAIP}, {1,1,1}, [=](group<3> grp)
    cgh.parallel_for_work_group(range<3>{NPT, numVPAIP, numVPAIP},  [=](group<3> grp)
    {
      const size_t pidx=grp[0];
      double *reconstructedPatch = Qout + sourcePatchSize*pidx;
      grp.parallel_for_work_item([&](auto idx)
      {
          const size_t x=idx.get_global_id(1);
          const size_t y=idx.get_global_id(2);
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          for (int i=0; i<unknowns+aux; i++)
          { 
            Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
          }
      });
    });
  }).wait();
 

  Q.submit([&](handler &cgh)
  {
    //sycl::stream out(100000, 768, cgh);
    cgh.parallel_for_work_group(range<3>{NPT, numVPAIP/2, numVPAIP},  [=](group<3> grp)
    //cgh.parallel_for_work_group(range<3>{NPT, numVPAIP/2, numVPAIP}, {1,1,1}, [=](group<3> grp)
    {
      const size_t pidx=grp[0];
      double *reconstructedPatch = Qout + sourcePatchSize*pidx;
      
      const double h0 = 0.1;
      const double dx = 0.1;//volumeH(0);


      for (int shift = 0; shift < 2; shift++)
      {

        // Normal 0
        grp.parallel_for_work_item([&](auto idx)
        {
            const int normal = 0;
            const size_t x=shift + 2*idx.get_global_id(1);
            const size_t y=idx.get_global_id(2);

            int leftVoxelInPreimage  = x +      (y + 1) * (2 + numVPAIP);
            int rightVoxelInPreimage = x + 1  + (y + 1) * (2 + numVPAIP);
            double * QL = reconstructedPatch + leftVoxelInPreimage  * (unknowns + aux);
            double * QR = reconstructedPatch + rightVoxelInPreimage * (unknowns + aux);
            
            double fluxFL[unknowns], fluxFR[unknowns], fluxNCP[unknowns];
                
            //if (not skipFluxEvaluation)
            //{
            flux(QL, normal, fluxFL);
            flux(QR, normal, fluxFR);
            //}

            double lambdaMaxL = maxEigenvalue(QL,normal);
            double lambdaMaxR = maxEigenvalue(QR,normal);
            double lambdaMax  = std::max( lambdaMaxL, lambdaMaxR );
            
            int leftVoxelInImage     = x - 1 + y * numVPAIP;
            int rightVoxelInImage    = x     + y * numVPAIP;
                
            for (int unknown = 0; unknown < unknowns; unknown++)
            {
              if (x > 0)
              {
                double fl = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                //if (not skipFluxEvaluation) 
                  fl +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown];
                Qout[pidx*destPatchSize + leftVoxelInImage * (unknowns + aux) + unknown]  -= dt / h0 * fl;
              }
              if (x < numVPAIP/2)
              {
                double fr = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                //if (not skipFluxEvaluation)
                  fr +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown]; 
                Qout[pidx*destPatchSize + rightVoxelInImage * (unknowns + aux) + unknown] += dt / h0 * fr;
              }
            }
        }); // Do we have an implicit barrier here or do we need to sync?
      
        // Normal 1 (NOTE: I simply swap x and y here, needs checked)
        grp.parallel_for_work_item([&](auto idx)
        {
            const int normal = 1;
            const size_t y=shift + 2*idx.get_global_id(1);
            const size_t x=idx.get_global_id(2);
           
            int lowerVoxelInPreimage = x + 1  +       y * (2 + numVPAIP);
            int upperVoxelInPreimage = x + 1  + (y + 1) * (2 + numVPAIP);
            int lowerVoxelInImage    = x      + (y - 1) *      numVPAIP ;
            int upperVoxelInImage    = x      +       y *      numVPAIP ;

            double* QL = reconstructedPatch + lowerVoxelInPreimage * (unknowns + aux);
            double* QR = reconstructedPatch + upperVoxelInPreimage * (unknowns + aux);

            
            double fluxFL[unknowns], fluxFR[unknowns], fluxNCP[unknowns];
                
            //if (not skipFluxEvaluation)
            //{
            flux(QL, normal, fluxFL);
            flux(QR, normal, fluxFR);
            //}

            double lambdaMaxL = maxEigenvalue(QL,normal);
            double lambdaMaxR = maxEigenvalue(QR,normal);
            double lambdaMax  = std::max( lambdaMaxL, lambdaMaxR );
                
            for (int unknown = 0; unknown < unknowns; unknown++)
            {
              if (y > 0)
              {
                double fl = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                //if (not skipFluxEvaluation) 
                  fl +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown];
                  Qout[pidx*destPatchSize + lowerVoxelInImage * (unknowns + aux) + unknown] -= dt / h0 * fl;
              }
              if (y < numVPAIP/2)
              {
                double fr = - 0.5 * lambdaMax * (QR[unknown] - QL[unknown]);
                //if (not skipFluxEvaluation)
                  fr +=   0.5 * fluxFL[unknown] + 0.5 * fluxFR[unknown]; 
                  Qout[pidx*destPatchSize + upperVoxelInImage * (unknowns + aux) + unknown] += dt / h0 * fr;
              }
            }
        }); // Do we have an implicit barrier here or do we need to sync?
      
      }
      });
  }).wait();
}


template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void qcompute(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
  const size_t NPT=20;
  const double dt =0.5;

  Q.submit([&](handler &cgh) 
  {
      //sycl::stream out(100000, 256, cgh);

      
      cgh.parallel_for(
      sycl::range<3>{NPT,numVPAIP, numVPAIP}, [=] (auto it) 
      { 
      
         const size_t pidx=it[0];
         const size_t x=it[1];
         const size_t y=it[2];
         double *reconstructedPatch = Qout + sourcePatchSize*pidx;
         
         int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
         int destinationIndex = y*numVPAIP + x;
         //out << "s: " << sourceIndex << " d: " << destinationIndex << "\n";
         //for (int i=0; i<unknowns+aux; i++) out << pidx*destPatchSize + destinationIndex*(unknowns+aux)+i << "\n";
         for (int i=0; i<unknowns+aux; i++)// out << sourceIndex*(unknowns+aux)+i << "\n";
         { 
           Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
         }

      });
  });
  Q.wait();
  
}


int main()
{
    queue Q(gpu_selector{});
    std::cout << "  Using SYCL device: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    const size_t NPT=20;
    const int numVPAIP = 17;
    const int srcPS = (numVPAIP+2)*(numVPAIP+2)*5;
    const int destPS = numVPAIP*numVPAIP*5;
    
    auto Xin  = malloc_shared<double>(srcPS*NPT, Q);
    for (int i=0;i<srcPS*NPT;i++) Xin[i]=i+1.2;
    auto Xout = malloc_shared<double>(destPS*NPT, Q);
   
    hcompute<17,5,0>(Q, 1, srcPS, destPS ,Xin, Xout);
}
