# cnn-bf
All of this work belongs to Đoàn Ý Nhi.
The source code was submitted as graduation thesis to School of Computer Science and Engineering, International University - Vietnam National University Ho Chi Minh City.
For more information, please read my thesis, Convolutional Neural Network of Bilateral Filter in Image Processing.

Prerequisite requirements:
1. Your device must include NVIDIA GPU(s).
2. Install CUDA Toolkit at https://developer.nvidia.com/cuda-downloads.
3. Install libraries cupy, numpy, and opencv.

Instructions
1. Compile and import this file.
2. Choose bilateral_filter_for_image to apply bilateral filter to image. Choose bilateral_filter_for_video to apply bilateral filter to video.<br>
    2.1. Format of bilateral_filter_for_image: bilateral_filter_for_image(path_in, path_out, kernel_size, sigma_s, sigma_i).<br>
        Default value for kernel_size=3, sigma_s=2.5, sigma_i=0.5, kernel_size must be an odd number.<br>
        E.g: bilateral_filter_for_image("D:\\Folder\\input_image.png", "D:\\Folder\\output_image.png", 5, 1.5, 0.25)<br>
    2.2. Format of bilateral_filter_for_video: bilateral_filter_for_video(path_in, path_out, kernel_size, sigma_s, sigma_i).<br>
        Default value for kernel_size=3, sigma_s=2.5, sigma_i=0.5, kernel_size must be an odd number.<br>
        E.g: bilateral_filter_for_video("D:\\Folder\\input_video.mp4", "D:\\Folder\\output_video.mp4", 5, 1.5, 0.25)<br>

Note:
1. I included a test image and a test video with the archived folder.
2. The output result is affected by the input values of parameters, and the performance of the hardware.
3. You can find all experimental results at http://bit.ly/ITITIU17025_Results. The experiment was done on NVIDIA GPU Tesla T4.
4. For more details, you can contact me via email: doanynhi1999@gmail.com

##Project description