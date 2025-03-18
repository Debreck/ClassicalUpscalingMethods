CC := g++
CUDA := nvcc

# Optimization flags
CPP_FLAGS := -Ofast -funroll-loops -finline-functions -march=native -mtune=native

# OpenMP flags
OMP_FLAGS := -fopenmp ${CPP_FLAGS}

# CUDA flags
CUDA_FLAGS := -std=c++11 -O3 -use_fast_math -maxrregcount=64

# Nearest neighbour interpolation versions
nearest_v1.0.0 := nearestNeigbour/nearestNeighbour_v1.0.0.cpp
nearest_v1.0.1 := nearestNeigbour/nearestNeighbour_v1.0.1.cpp
nearest_v1.1.0 := nearestNeigbour/nearestNeighbour_v1.1.0.cpp
nearest_v1.1.1 := nearestNeigbour/nearestNeighbour_v1.1.1.cpp
nearest_v1.1.2 := nearestNeigbour/nearestNeighbour_v1.1.2.cpp
nearest_v1.2.0 := nearestNeigbour/nearestNeighbour_v1.2.0.cu

# Bilinear interpolation versions
bilinear_v1.0.0 := bilinearInterpolation/bilinear_v1.0.0.cpp
bilinear_v1.0.1 := bilinearInterpolation/bilinear_v1.0.1.cpp
bilinear_v1.1.0 := bilinearInterpolation/bilinear_v1.1.0.cpp
bilinear_v1.1.1 := bilinearInterpolation/bilinear_v1.1.1.cpp
bilinear_v1.2.0 := bilinearInterpolation/bilinear_v1.2.0.cu

# Lanczos resampler versions
lanczos_v1.0.0 := lanczosResampling/lanczos_v1.0.0.cpp
lanczos_v1.0.1 := lanczosResampling/lanczos_v1.0.1.cpp
lanczos_v1.1.0 := lanczosResampling/lanczos_v1.1.0.cpp
lanczos_v1.1.1 := lanczosResampling/lanczos_v1.1.1.cpp

nearest: nearest_v1.0.0 nearest_v1.0.1 nearest_v1.1.0 nearest_v1.1.1 nearest_v1.1.2 nearest_v1.2.0

bilinear: bilinear_v1.0.0 bilinear_v1.0.1 bilinear_v1.1.0 bilinear_v1.1.1 bilinear_v1.2.0

lanczos: lanczos_v1.0.0 lanczos_v1.0.1 lanczos_v1.1.0 lanczos_v1.1.1

all: nearest bilinear lanczos

nearest_v1.0.0: ${nearest_v1.0.0}
	${CC} ${CPP_FLAGS} -o nearest_upscaler_v1.0.0.exe ${nearest_v1.0.0}

nearest_v1.0.1: ${nearest_v1.0.1}
	${CC} ${CPP_FLAGS} -o nearest_upscaler_v1.0.1.exe ${nearest_v1.0.1}

nearest_v1.1.0: ${nearest_v1.1.0}
	${CC} ${OMP_FLAGS} -o nearest_upscaler_v1.1.0.exe ${nearest_v1.1.0}

nearest_v1.1.1: ${nearest_v1.1.1}
	${CC} ${OMP_FLAGS} -o nearest_upscaler_v1.1.1.exe ${nearest_v1.1.1}

nearest_v1.1.2: ${nearest_v1.1.2}
	${CC} ${OMP_FLAGS} -o nearest_upscaler_v1.1.2.exe ${nearest_v1.1.2}

nearest_v1.2.0: ${nearest_v1.2.0}
	${CUDA} ${CUDA_FLAGS} -o nearest_upscaler_v1.2.0.exe ${nearest_v1.2.0}


bilinear_v1.0.0: ${bilinear_v1.0.0}
	${CC} ${CPP_FLAGS} -o bilinear_upscaler_v1.0.0.exe ${bilinear_v1.0.0}

bilinear_v1.0.1: ${bilinear_v1.0.1}
	${CC} ${CPP_FLAGS} -o bilinear_upscaler_v1.0.1.exe ${bilinear_v1.0.1}

bilinear_v1.1.0: ${bilinear_v1.1.0}
	${CC} ${OMP_FLAGS} -o bilinear_upscaler_v1.1.0.exe ${bilinear_v1.1.0}

bilinear_v1.1.1: ${bilinear_v1.1.1}
	${CC} ${OMP_FLAGS} -o bilinear_upscaler_v1.1.1.exe ${bilinear_v1.1.1}

bilinear_v1.2.0: ${bilinear_v1.2.0}
	${CUDA} ${CUDA_FLAGS} -o bilinear_upscaler_v1.2.0.exe ${bilinear_v1.2.0}


lanczos_v1.0.0: ${lanczos_v1.0.0}
	${CC} ${CPP_FLAGS} -o lanczos_upscaler_v1.0.0.exe ${lanczos_v1.0.0}

lanczos_v1.0.1: ${lanczos_v1.0.1}
	${CC} ${CPP_FLAGS} -o lanczos_upscaler_v1.0.1.exe ${lanczos_v1.0.1}

lanczos_v1.1.0: ${lanczos_v1.1.0}
	${CC} ${OMP_FLAGS} -o lanczos_upscaler_v1.1.0.exe ${lanczos_v1.1.0}

lanczos_v1.1.1: ${lanczos_v1.1.1}
	${CC} ${OMP_FLAGS} -o lanczos_upscaler_v1.1.1.exe ${lanczos_v1.1.1}

clean:
	@rm -f *_upscaler_v*.exe