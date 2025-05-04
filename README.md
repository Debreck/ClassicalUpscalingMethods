# Classical Image Upscaling Methods
Implementation of 3 classical image upscalers -- Nearest Neigbour, Bilinear Interpolation and Lanczos Resampling.

Disclaimer: This repository was created as part of a master's thesis at CTU FIT -- Comparison and implementation of classical image upscaling methods.
Licence: MIT type (Free licence)

## Version list
### Nearest Neighbour
- **v1.0.0**  - Naive sequential implementation
- **v1.0.1**  - Naive sequential implementation without std::vector<>
- **v1.1.0**  - Naive parallel implementation with OpenMP (*previous v1.0.0*)
- **v1.1.1**  - Naive parallel implementation with OpenMP (*previous v1.0.1*)
- **v1.1.2**  - OpenMP optimalizations with looptiling (*previous v1.1.1*)
- **v1.2.0**  - CUDA implementation (*previous v1.0.1*)

### Bilinear Interpolation
- **v1.0.0**  - Naive sequential implementation
- **v1.0.1**  - Sequential implementation optimalizations and fixes
- **v1.1.0**  - Naive parallel implementation with OpenMP
- **v1.1.1**  - OpenMP optimalizations with looptiling
- **v1.2.0**  - CUDA implementation (*previous v1.0.1*)

### Lanczos Resampling
- **v1.0.0**  - Naive sequential implementation
- **v1.0.1**  - Sequential implementation with kernel precalculation optimalization
- **v1.1.0**  - Naive parallel implementation with OpenMP
- **v1.1.1**  - OpenMP optimalizations with looptiling
- **v1.2.0**  - CUDA implementation (*previous v1.0.0*)


## General usage
- 1 input argument: `upsaler.exe <input_pic_name>.ppm`
- 2 input arguments: `upsaler.exe <input_pic_name>.ppm <output_pic_name>.ppm`
- 4 input arguments: `upsaler.exe <wanted_width> <wanted_height> <input_pic_name>.ppm <output_pic_name>.ppm`
- 5 input arguments: `upsaler.exe <wanted_width> <wanted_height> <input_pic_name>.ppm <output_pic_name_NOT_PRINTED>.ppm <boolean_for_only_time_output>`
