#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>
#include <cmath>

// Definition of mathematical constant Pi
const float pi = 3.1415926536;

// Parameter "a" determines size of Lanczos kernel
const int a = 3;

// Predefined clamp(...) function
int clamp( const int x, const int y );

// Structure for sizes needing width and height
struct SSize
{
    public:
        int _width  = 0;
        int _height = 0;

    int total() const
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
        CPixel( const int red, const int green, const int blue )
            : _red( red ), _green( green ), _blue( blue ) {}

        int red()   const { return _red;   }
        int green() const { return _green; }
        int blue()  const { return _blue;  }

        CPixel& operator+= ( const CPixel& otherPixel )
        {
            _red += otherPixel._red;
            _green += otherPixel._green;
            _blue += otherPixel._blue;
            return *this;
        }
        
        void clampRGB()
        {
            _red   = clamp( _red,   255 );
            _green = clamp( _green, 255 );
            _blue  = clamp( _blue,  255 );
        }
};

inline CPixel operator* ( const CPixel& pixel, const float num )
{
    return CPixel( float( pixel.red() ) * num,
                   float( pixel.green() ) * num,
                   float( pixel.blue() ) * num );
}

inline CPixel operator* ( const float num, const CPixel &pixel)
{
    return pixel * num;
}

inline std::ostream& operator<< ( std::ostream &out, const CPixel &pixel )
{ 
    return out << pixel.red() << ' ' << pixel.green() << ' ' << pixel.blue();
}

// Helper function for clamping -- 0 < x < y
int clamp( const int x, const int y )
{
    if( x < 0 ) return 0;
    if( x > y ) return y;
    return x;
}

// Helper function to load a picture from a file
bool loadPicture( const std::string fileName, CPixel* &picture, SSize& picSize )
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
bool savePicture( const std::string fileName, const CPixel* picture, const SSize& picSize )
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
bool toIntSafely( int& toInt, const char* charToInt )
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
bool argParsing( int argc, char** argv, SSize& newPicSize, std::string& inputFileName, std::string& outputFileName, bool& multiRun )
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

// Lanczos Kernel function
float lanczosKernel( const float x )
{
    if( x == 0 ) return 1;
    else if( -1*a < x && x < a )
    {
        return (a * std::sin( pi * x ) * std::sin( (pi * x) / a )) / ( std::pow( pi, 2 ) * std::pow( x, 2 ) );
    }

    return 0;
}

// Lanczos Kernel precalculation helper function
void kernelPrecalculation( std::map<float,float> &kernels, const float x )
{
    int sumFrom = int(x) - a + 1;
    int sumTo   = int(x) + a;
    for( int idx = sumFrom; idx <= sumTo; ++idx )
    {
        kernels[x - float(idx)] = lanczosKernel( x - idx );
    }
}

// Lanczos Resampling helper function -- resampling horizontally
CPixel resamplingH( const float x, const CPixel* oldPicture, const std::map<float,float> &kernels, const size_t oldWidth, const size_t moveIdx )
{
    int sumFrom = int(x) - a + 1;
    int sumTo   = int(x) + a;
    CPixel sum  = {};
    for( int idx = sumFrom; idx <= sumTo; ++idx )
    {
        sum += oldPicture[clamp( idx, oldWidth - 1 ) + moveIdx] * kernels.at(x - idx);
    }

    sum.clampRGB();
    return sum;
}

// Lanczos Resampling helper function -- resampling vertically
CPixel resamplingV( const float x, const CPixel* interPicture, const std::map<float,float> &kernels, const int oldPicHeight, const int newPicWidth, const size_t moveIdx )
{
    int sumFrom = int(x) - a + 1;
    int sumTo   = int(x) + a;
    CPixel sum  = {};
    for( int idx = sumFrom; idx <= sumTo; ++idx )
    {
        sum += interPicture[( clamp( idx, oldPicHeight - 1 ) * newPicWidth ) + moveIdx] * kernels.at(x - idx);
    }

    sum.clampRGB();
    return sum;
}

// Lanczos Resampling function
void lanczosResampling( const CPixel* __restrict__ oldPicture, CPixel* __restrict__ newPicture, const SSize oldPicSize, const SSize newPicSize )
{
    std::map<float,float> calcKernels;
    int idx;
    
    const float widthRatio    = float(oldPicSize._width)  / float(newPicSize._width);
    const float heightRatio   = float(oldPicSize._height) / float(newPicSize._height);
    const int biggerDimension = newPicSize._width > newPicSize._height ? newPicSize._width : newPicSize._height;
    const int picInterSize    = newPicSize._width*oldPicSize._height;
    const int picTotalSize    = newPicSize._width*newPicSize._height;
    const int cacheAlignment  = 64;

    CPixel* interPicture = new CPixel[picInterSize];

    for( int idx = 0; idx < biggerDimension; ++idx )
    {
        kernelPrecalculation( calcKernels, (idx + 0.5) * widthRatio  - 0.5 );
        kernelPrecalculation( calcKernels, (idx + 0.5) * heightRatio - 0.5 );
    }

    #pragma omp parallel
    {
        #pragma omp for
        for( idx = 0; idx < picInterSize + cacheAlignment - 1; idx += cacheAlignment )
        {
            for( int newPicIdx = idx; newPicIdx < (picInterSize > idx+cacheAlignment ? idx+cacheAlignment : picInterSize); ++newPicIdx )
            {
                const float x = (float(newPicIdx%newPicSize._width) + 0.5) * widthRatio - 0.5;
                interPicture[newPicIdx] = resamplingH( x, oldPicture, calcKernels, oldPicSize._width, (newPicIdx/newPicSize._width)*oldPicSize._width );
            }
        }

        #pragma omp for
        for( idx = 0; idx < picTotalSize + cacheAlignment - 1; idx += cacheAlignment )
        {
            for( int newPicIdx = idx; newPicIdx < (picTotalSize > idx+cacheAlignment ? idx+cacheAlignment : picTotalSize); ++newPicIdx )
            {
                const float x = (float(newPicIdx/newPicSize._width) + 0.5) * heightRatio - 0.5;
                newPicture[newPicIdx] = resamplingV( x, interPicture, calcKernels, oldPicSize._height, newPicSize._width, newPicIdx%newPicSize._width );
            }
        }
    }
    

    delete[] interPicture;
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

    if( !multiRun )
    {
        std::cout << "DEBUG: From: " << std::setw( 4 ) << oldPicSize._width << 'x' << oldPicSize._height << '\n';
        std::cout << "       To:   " << std::setw( 4 ) << newPicSize._width << 'x' << newPicSize._height << std::endl;
    }

    auto startTimer = std::chrono::high_resolution_clock::now();

    lanczosResampling( oldPicture, newPicture, oldPicSize, newPicSize );

    auto endTimer = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> ( endTimer - startTimer );
    if( !multiRun )
    {
        std::cout << "Total render time: " << int(duration.count()) / 1000000 << "s " << ( int(duration.count()) / 1000 ) % 1000 << "ms " << int(duration.count()) % 1000 << "µs" << std::endl;
    }
    else
    {
        std::cout << int(duration.count()) << std::endl;
    }
    
    if( !multiRun ) savePicture( outputFileName, newPicture, newPicSize );

    delete[] oldPicture;
    delete[] newPicture;
    return 0;
}
