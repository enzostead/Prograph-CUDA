#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

/**
 * Kernel pour transformer l'image RGB en niveaux de gris.
 */
__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }
}

/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris.
 */
__global__ void lg( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows )
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 2 && i < cols && j > 2 && j < rows )
  {
    // Edge detection
	auto e = 
		- g[((j - 2) * cols + i)]
		- g[((j - 1) * cols + i - 1)] - 2 * g[((j -1) * cols + i)] - g[((j - 1) * cols + i + 1)]
		- g[(j * cols + i - 2)] - 2 * g[(j * cols + i - 1)] + 16 * g[(j * cols + i)] - 2 * g[(j * cols + i + 1)] - g[(j * cols + i + 2)]
		- g[((j + 1) * cols + i - 1)] - 2 * g[((j + 1) * cols + i + 1)] - g[((j + 1) * cols + i)]
		- g[((j + 2) * cols + i)];
	   
    e = e > 255 ? e = 255 : e;
    e = e < 0 ? e = 0 : e;

    s[ j * cols + i ] = e;
  }
}


/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris, en utilisant la mémoire shared
 * pour limiter les accès à la mémoire globale.
 */
__global__ void lg_shared( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows )
{
  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  auto i = blockIdx.x * (blockDim.x-5) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-5) + threadIdx.y;

  extern __shared__ unsigned char sh[];

  if( i < cols && j < rows )
  {
    sh[ lj * w + li ] = g[ j * cols + i ];
  }

  __syncthreads();

  if( i < cols -2 && j < rows-2 && li > 0 && li < (w-2) && lj > 0 && lj < (h-2) )
  {
    // Edge detection
	auto e = 
		- sh[((lj - 2) * w + li)]
		- sh[((lj - 1) * w + li - 1)] - 2 * sh[((lj -1) * w + li)] - sh[((lj - 1) * w + li + 1)]
		- sh[(lj * w + li - 2)] - 2 * sh[(lj * w + li - 1)] + 16 * sh[(lj * w + li)] - 2 * sh[(lj * w + li + 1)] - sh[(lj * w + li + 2)]
		- sh[((lj + 1) * w + li - 1)] - 2 * sh[((lj + 1) * w + li + 1)] - sh[((lj + 1) * w + li)]
		- sh[((lj + 2) * w + li)];
   
    e = e > 255 ? e = 255 : e;
    e = e < 0 ? e = 0 : e;

    s[ j * cols + i ] = e;
  }
}


/**
 * Kernel fusionnant le passage en niveaux de gris et la détection de contours.
 */
__global__ void grayscale_lg_shared( unsigned char * rgb, unsigned char * s, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * (blockDim.x-5) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-5) + threadIdx.y;

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
 
  if( i < cols - 2 && j < rows-2 && li > 2 && li < (w-2) && lj > 2 && lj < (h-2) )
  {
	auto e = 
		- sh[((lj - 2) * w + li)]
		- sh[((lj - 1) * w + li - 1)] - 2 * sh[((lj -1) * w + li)] - sh[((lj - 1) * w + li + 1)]
		- sh[(lj * w + li - 2)] - 2 * sh[(lj * w + li - 1)] + 16 * sh[(lj * w + li)] - 2 * sh[(lj * w + li + 1)] - sh[(lj * w + li + 2)]
		- sh[((lj + 1) * w + li - 1)] - 2 * sh[((lj + 1) * w + li + 1)] - sh[((lj + 1) * w + li)]
		- sh[((lj + 2) * w + li)];
   
    e = e > 255 ? e = 255 : e;
    e = e < 0 ? e = 0 : e;

    s[ j * cols + i ] = e;
  }
}


int main()
{
  cv::Mat m_in = cv::imread("in2.jpg", cv::IMREAD_UNCHANGED );

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
  dim3 grid0( ( cols - 2) / block.x + 1 , ( rows - 2 ) / block.y + 1 );
  /**
   * Pour la version shared il faut faire superposer les blocs de 2 pixels
   * pour ne pas avoir de bandes non calculées autour des blocs
   * on crée donc plus de blocs.
   */
  dim3 grid1((cols - 2) / (block.x - 5)+1, (rows - 2) / (block.y - 5)+1);

    
  cudaEvent_t start, stop;

  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  // Mesure du temps de calcul du kernel uniquement.
  cudaEventRecord( start );


  /*
  // Version en 2 étapes.
  grayscale<<< grid0, block >>>( rgb_d, g_d, cols, rows );
  lg<<< grid0, block >>>( g_d, s_d, cols, rows );
  */
  
  
  
  /*
  // Version en 2 étapes, lg avec mémoire shared.
  grayscale<<< grid0, block >>>( rgb_d, g_d, cols, rows );
  lg_shared<<< grid1, block, block.x * block.y >>>( g_d, s_d, cols, rows );
  */

  // Version fusionnée.
  grayscale_lg_shared<<< grid1, block, block.x * block.y >>>( rgb_d, s_d, cols, rows );
 
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
  std::cout << "time=" << duration <<"ms"<< std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cv::imwrite( "out_laplaciang_cu.jpg", m_out );

  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaFree( s_d);

  cudaFreeHost( g );
  cudaFreeHost( rgb );
  
  return 0;
}
