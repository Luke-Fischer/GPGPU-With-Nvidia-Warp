# GPGPU-With-Nvidia-Warp
Using the Nvidia Warp API to sharpen and denoise greyscale and coloured images

# Usage
python3 process_image.py algType kernelSize param inFileName outFileName
where:
  - algType is "-s" to sharpen or "-n" to denoise
  - kernelSize is the size of the kernel (i.e 3x3, 5x5, etc... must be positive and odd)
  - param is the numerical parameter the unmasking algorithm needs (effective values range from 0.2 - 0.9)
  - inFileName is the image you want to process
  - outFileName is the processed image

# My Approach
- To sharpen an image my program is using an unmask sharpening technique with mean filtering
- To denoise an image my program is using median filtering
