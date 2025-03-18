#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>

// Maximum number of threads per block in a CUDA kernel
#define THREADS_PER_BLOCK_LIMIT 1024

// Error handling macro for CUDA library
#define ERROR_HANDLER( err ) ( HandleError( err, __FILE__, __LINE__ ) )
static void HandleError( cudaError_t err, const char* file, int line )
{
    if( err != cudaSuccess )
    {
        fprintf( stderr, "ERROR: %s in %s at a line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

// Predefined clamp(...) function
__device__ int clampInt( const int x, const int y );

// Structure for sizes needing width and height
struct SSize
{
    public:
        int _width  = 0;
        int _height = 0;

        __host__ __device__ int total() const
    {
        return _width * _height;
    }
};

// Class for RGB pixels
class CPixel
{
    private:
        int _red   = 0;
        int _green = 0;
        int _blue  = 0;

    public:
        CPixel() = default;
        __host__ __device__ CPixel( const int red, const int green, const int blue )
            : _red( red ), _green( green ), _blue( blue ) {}

        __host__ __device__ int red()   const { return _red;   }
        __host__ __device__ int green() const { return _green; }
        __host__ __device__ int blue()  const { return _blue;  }

        __device__ CPixel& operator+= ( const CPixel& otherPixel )
        {
            _red += otherPixel._red;
            _green += otherPixel._green;
            _blue += otherPixel._blue;
            return *this;
        }
        
        __device__ void clampRGB()
        {
            _red   = clampInt( _red,   255 );
            _green = clampInt( _green, 255 );
            _blue  = clampInt( _blue,  255 );
        }
};

__device__ inline CPixel operator* ( const CPixel &pixel, const float num )
{
    return CPixel( float( pixel.red() ) * num,
                   float( pixel.green() ) * num,
                   float( pixel.blue() ) * num );
}

__device__ inline CPixel operator* ( const float num, const CPixel &pixel)
{
    return pixel * num;
}

__device__ inline CPixel operator+ ( const CPixel &pixel, const CPixel& otherPixel )
{
    return CPixel( pixel.red()   + otherPixel.red(),
                   pixel.green() + otherPixel.green(),
                   pixel.blue()  + otherPixel.blue() );
}

__host__ inline std::ostream& operator<< ( std::ostream &out, const CPixel &pixel )
{ 
    return out << pixel.red() << ' ' << pixel.green() << ' ' << pixel.blue();
}

// Helper function for clamping -- 0 < x < y
__device__ int clampInt( const int x, const int y )
{
    if( x < 0 ) return 0;
    if( x > y ) return y;
    return x;
}

// Helper function for clamping -- 0 < x < y
__device__ float clampfloat( const float x, const float y )
{
    if( x < 0 ) return 0;
    if( x > y ) return y;
    return x;
}

// Helper function to load a picture from a file
__host__ bool loadPicture( const std::string fileName, CPixel* &picture, SSize& picSize )
{
    std::ifstream inputFile( fileName, std::ios::in );
    if( ! inputFile.is_open() )
    {
        std::cerr << "Unable to open the file \"" << fileName << "\"!" << std::endl;
        return false;
    }

    std::string tmpStr;
    inputFile >> tmpStr;
    if( tmpStr != "P3" )
    {
        std::cerr << "Bad format of the file \"" << fileName << "\"! -- MAGIC_NUM" << std::endl;
        inputFile.close();
        return false;
    }

    inputFile >> picSize._width;
    inputFile >> picSize._height;
    try
    {
        picture = new CPixel[picSize.total()];
    }
    catch( std::bad_alloc const& err )
    {
        std::cerr << "std::bad_alloc::what(): " << err.what() << '\n';
        std::cerr << "Unable to allocate enought space for the input picture!" << std::endl;
        return false;
    }

    inputFile >> tmpStr;
    if( tmpStr != "255" )
    {
        std::cerr << "Bad format of the file \"" << fileName << "\"! -- MAX_VAL" << std::endl;
        inputFile.close();
        return false;
    }
    int red, green, blue;
    for( int idx = 0; idx < picSize.total(); ++idx )
    {
        inputFile >> red;
        // std::cout << red << std::endl;
        inputFile >> green;
        // std::cout << green << std::endl;
        inputFile >> blue;
        // std::cout << blue << std::endl;
        
        picture[idx] = CPixel( red, green, blue );
    }

    inputFile.close();
    return true;
}

// Helper function to save a resampled picture to a file
__host__ bool savePicture( const std::string fileName, const CPixel* picture, const SSize& picSize )
{
    std::ofstream pictureFile( std::string(fileName), std::ios::out | std::ios::trunc );
    if( ! pictureFile.is_open() )
    {
        std::cout << "Unable to open the file \"" << fileName << "\" to print!" << std::endl;
        return false;
    }

    std::cout << "Printing a picture..." << std::endl;
    pictureFile << "P3\n" << picSize._width << ' ' << picSize._height << "\n255\n";
    for( int idx = 0; idx < picSize.total(); ++idx )
    {
        pictureFile << picture[idx] << '\n';
    }

    pictureFile.close();
    std::cout << std::endl;
    return true;

}

// Safely try to parse characters into the integers
__host__ bool toIntSafely( int& toInt, const char* charToInt )
{
    try
    {
        toInt = std::stoi( charToInt );
    }
    catch( std::invalid_argument const& err )
    {
        std::cerr << "std::invalid_argument::what(): " << err.what() << std::endl;
        return false;
    }
    catch( std::out_of_range const& err )
    {
        std::cerr << "std::out_of_range::what(): " << err.what() << std::endl;
        return false;
    }

    return true;
}

// Helper function to parse intput arguments
__host__ bool argParsing( int argc, char** argv, SSize& newPicSize, std::string& inputFileName, std::string& outputFileName, bool& multiRun )
{
    if( argc == 6 )
    {
        if( !toIntSafely( newPicSize._width,  argv[1] ) ) return false;
        if( !toIntSafely( newPicSize._height, argv[2] ) ) return false;

        inputFileName  = argv[3];
        outputFileName = argv[4];
        multiRun       = bool( argv[5] );
    }
    else if( argc == 5 )
    {
        if( !toIntSafely( newPicSize._width,  argv[1] ) ) return false;
        if( !toIntSafely( newPicSize._height, argv[2] ) ) return false;
        
        inputFileName  = argv[3];
        outputFileName = argv[4];
    }
    else if( argc == 3 )
    {
        newPicSize._width  = 124;
        newPicSize._height = 124;
        inputFileName      = argv[1];
        outputFileName     = argv[2];
    }
    else if( argc == 2 )
    {
        newPicSize._width  = 124;
        newPicSize._height = 124;
        inputFileName      = argv[1];
        outputFileName     = "pic_out.ppm";
    }
    else
    {
        return false;
    }

    return true;
}

// Function for Bilinear Interpolation in 2D pictures with RGB pixels
__global__ void bilinearInterpolation( const CPixel* oldPicture, CPixel* newPicture, const SSize* oldPicSize, const SSize* newPicSize )
{
    const float widthRatio      = float(oldPicSize->_width)  / float(newPicSize->_width);
    const float heightRatio     = float(oldPicSize->_height) / float(newPicSize->_height);
    const float oldPicWidthIdx  = oldPicSize->_width  - 1;
    const float oldPicHeightIdx = oldPicSize->_height - 1;

    const int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if( idx >= newPicSize->total() ) return;

    const float x = ((float(idx % newPicSize->_width) + 0.5) * widthRatio)  - 0.5;
    const float y = ((float(idx / newPicSize->_width) + 0.5) * heightRatio) - 0.5;
    
    const int xCeil = std::ceil( x );
    const int yCeil = std::ceil( y );

    const int x1 = x;
    const int x2 = xCeil < oldPicWidthIdx ? xCeil : oldPicWidthIdx;
    const int y1 = y;
    const int y2 = yCeil < oldPicHeightIdx ? yCeil : oldPicHeightIdx;

    if( x1 == x2 && y1 == y2 )
    {
        newPicture[idx] = oldPicture[int(y) * oldPicSize->_width + int(x)];
    }
    else if( x1 == x2 )
    {
        const CPixel R1 = oldPicture[y1 * oldPicSize->_width + int(x)];
        const CPixel R2 = oldPicture[y2 * oldPicSize->_width + int(x)];

        newPicture[idx] = R1 * (y2 - y) + R2 * (y - y1);
    }
    else if( y1 == y2 )
    {
        const CPixel R1 = oldPicture[int(y) * oldPicSize->_width + x1];
        const CPixel R2 = oldPicture[int(y) * oldPicSize->_width + x2];

        newPicture[idx] = R1 * (x2 - x) + R2 * (x - x1);
    }
    else
    {
        const int idxQ11 = y1 * oldPicSize->_width + x1;
        const int idxQ12 = idxQ11 + 1;
        const int idxQ21 = y2 * oldPicSize->_width + x1;
        const int idxQ22 = idxQ21 + 1;

        const CPixel R1 = float(x2 - x) / float(x2 - x1) * oldPicture[idxQ11] + float(x - x1) / float(x2 - x1) * oldPicture[idxQ12];
        const CPixel R2 = float(x2 - x) / float(x2 - x1) * oldPicture[idxQ21] + float(x - x1) / float(x2 - x1) * oldPicture[idxQ22];
        newPicture[idx] = float(y2 - y) / float(y2 - y1) * R1 + float(y - y1) / float(y2 - y1) * R2;
    }
}

int main( int argc, char** argv )
{
    SSize       newPicSize;
    std::string inputFileName, outputFileName;
    bool        multiRun = false;

    if( !argParsing( argc, argv, newPicSize, inputFileName, outputFileName, multiRun ) )
    {
        std::cerr << "Unable to parse the input arguments! Maybe a wrong number of input arguments." << std::endl;
        return 1;
    }
    
    SSize   oldPicSize;
    CPixel* oldPicture = nullptr;
    CPixel* newPicture = nullptr;
    if( !loadPicture( inputFileName, oldPicture, oldPicSize ) )
    {
        std::cerr << "Unable to load the picture!" << std::endl;
        return 2;
    }

    SSize* gpuNewPicSize;
    cudaMalloc( (void**)&gpuNewPicSize, sizeof( SSize ) );
    ERROR_HANDLER( cudaGetLastError() );
    
    SSize* gpuOldPicSize;
    cudaMalloc( (void**)&gpuOldPicSize, sizeof( SSize ) );
    ERROR_HANDLER( cudaGetLastError() );

    CPixel* gpuNewPicture;
    cudaMalloc( (void**)&gpuNewPicture, newPicSize.total()*sizeof( CPixel ) );
    ERROR_HANDLER( cudaGetLastError() );

    CPixel* gpuOldPicture;
    cudaMalloc( (void**)&gpuOldPicture, oldPicSize.total()*sizeof( CPixel ) );
    ERROR_HANDLER( cudaGetLastError() );

    cudaMemcpy( gpuNewPicSize, &newPicSize, sizeof( SSize ), cudaMemcpyHostToDevice );
    ERROR_HANDLER( cudaGetLastError() );
    cudaMemcpy( gpuOldPicSize, &oldPicSize, sizeof( SSize ), cudaMemcpyHostToDevice );
    ERROR_HANDLER( cudaGetLastError() );
    cudaMemcpy( gpuOldPicture, oldPicture, oldPicSize.total()*sizeof( CPixel ), cudaMemcpyHostToDevice );
    ERROR_HANDLER( cudaGetLastError() );
    cudaMemcpy( gpuOldPicture, oldPicture, oldPicSize.total()*sizeof( CPixel ), cudaMemcpyHostToDevice );
    ERROR_HANDLER( cudaGetLastError() );
    delete[] oldPicture;

    if( !multiRun )
    {
        std::cout << "DEBUG: From: " << std::setw( 4 ) << oldPicSize._width << 'x' << oldPicSize._height << '\n';
        std::cout << "       To:   " << std::setw( 4 ) << newPicSize._width << 'x' << newPicSize._height << std::endl;
    }

    size_t blockCount = (newPicSize.total() / THREADS_PER_BLOCK_LIMIT) + 1;
    auto startTimer = std::chrono::high_resolution_clock::now();

    bilinearInterpolation<<<blockCount, THREADS_PER_BLOCK_LIMIT>>>( gpuOldPicture, gpuNewPicture, gpuOldPicSize, gpuNewPicSize );
    cudaDeviceSynchronize();
    ERROR_HANDLER( cudaGetLastError() );

    auto endTimer = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> ( endTimer - startTimer );
    if( !multiRun )
    {
        std::cout << "Total render time: " << int(duration.count()) / 60000 << "m " << ( int(duration.count()) / 1000 ) % 60 << "s " << int(duration.count()) % 1000 << "ms" << std::endl;
    }
    else
    {
        std::cout << int(duration.count()) << std::endl;
    }

    cudaFree( gpuOldPicture );
    
    try
    {
        newPicture = new CPixel[newPicSize.total()];
    }
    catch( std::bad_alloc const& err )
    {
        std::cerr << "std::bad_alloc::what(): " << err.what() << '\n';
        std::cerr << "Unable to allocate enought space for the new picture!" << std::endl;
        return 3;
    }

    cudaMemcpy( newPicture, gpuNewPicture, newPicSize.total()*sizeof( CPixel ), cudaMemcpyDeviceToHost );
    ERROR_HANDLER( cudaGetLastError() );

    if( !multiRun ) savePicture( outputFileName, newPicture, newPicSize );

    delete[] newPicture;
    cudaFree( gpuNewPicture );
    cudaFree( gpuOldPicSize );
    cudaFree( gpuNewPicSize );

    return 0;
}
