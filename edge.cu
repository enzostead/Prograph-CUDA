#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

/**
 * Kernel fusionnant le passage en niveaux de gris et la détection de contours.
 */
__global__ void grayscale_edge_shared( unsigned char * rgb, unsigned char * s, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  extern __shared__ unsigned char sh[];

  if( i < cols && j < rows ) {
    sh[ lj * w + li ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }

  /**
   * Il faut synchroniser tous les warps (threads) du bloc pour être certain que le niveau de gris est calculé
   * par tous les threads du bloc avant de pouvoir accéder aux données des pixels voisins.
   */
  __syncthreads();
 
  if( i < cols -1 && j < rows-1 && li > 0 && li < (w-1) && lj > 0 && lj < (h-1) )
  {
      
   
    // Edge detection
    auto e = 
        - sh[((lj - 1) * w + li - 1)] - sh[((lj -1) * w + li)] - sh[((lj - 1) * w + li + 1)]
        - sh[(lj * w + li - 1)] + 8 * sh[(lj * w + li)] - sh[(lj * w + li + 1)]  
        - sh[((lj + 1) * w + li - 1)] - sh[((lj + 1) * w + li + 1)] - sh[((lj + 1) * w + li)];
        
    e = e > 255 ? 255 : e;
    e = e < 0 ? 0 : e;
    
    s[ j * cols + i ] = e;
  }
}


int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );

  //auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  //std::vector< unsigned char > g( rows * cols );
  // Allocation de l'image de sortie en RAM côté CPU.
  unsigned char * g = nullptr;
  cudaMallocHost( &g, rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g );

  // Copie de l'image en entrée dans une mémoire dite "pinned" de manière à accélérer les transferts.
  // OpenCV alloue la mémoire en interne lors de la décompression de l'image donc soit sans doute avec
  // un malloc standard.
  unsigned char * rgb = nullptr;
  cudaMallocHost( &rgb, 3 * rows * cols );
  
  std::memcpy( rgb, m_in.data, 3 * rows * cols );

  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * s_d;

  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &s_d, rows * cols );

  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  dim3 block( 64, 8 );
  dim3 grid0( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );
  /**
   * Pour la version shared il faut faire superposer les blocs de 2 pixels
   * pour ne pas avoir de bandes non calculées autour des blocs
   * on crée donc plus de blocs.
   */
  dim3 grid1( ( cols - 1) / (block.x-2) + 1 , ( rows - 1 ) / (block.y-2) + 1 );
    
  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel uniquement.
  cudaEventRecord( start );

  // Version fusionnée.
  grayscale_edge_shared<<< grid1, block, block.x * block.y >>>( rgb_d, s_d, cols, rows );
  
  // Vérification des erreurs
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
      printf("Erreur CUDA: %s\n", cudaGetErrorString(error));
  }

  cudaEventRecord( stop );
  
  cudaMemcpy( g, s_d, rows * cols, cudaMemcpyDeviceToHost );

  cudaEventSynchronize( stop );
  float duration;
  cudaEventElapsedTime( &duration, start, stop );
  std::cout << "time=" << duration <<"ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "out_edge_cu.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaFree( s_d);

  cudaFreeHost( g );
  cudaFreeHost( rgb );
  
  return 0;
}
