# Import libraries
import cv2
import cupy as cp
import numpy as np
import math


#Space distance kernel
def calculate_distance_kernel(size):
  kernel = cp.zeros([size, size])
  center = size // 2
  for x in range(size):
    for y in range(size):
      kernel[x][y] = (x - center) ** 2 + (y - center) ** 2
  return kernel

def calculate_norm_distance_kernel(size):
  kernel = cp.zeros((size, size))
  center = size // 2
  for x in range(size):
    for y in range(size):
      kernel[x][y] = cp.sqrt((x - center) ** 2 + (y - center) ** 2)
  return kernel

#Gaussian function
def gaussian_function(window, sigma):
  window = cp.asarray(window)
  sigma = cp.float32(sigma)
  out = cp.exp(-cp.square(window) / (2 * cp.square(sigma)))
  a = 1 / (math.sqrt(2 * math.pi) * sigma)
  return a * out

#Convolution
def n_conv(c, k):
  height, width = c.shape
  # print(k.shape)
  ele, h_k, w_k = k.shape
  pad = h_k // 2
  # ele = h_k * w_k
  kernel = cp.zeros((ele, ele, 1, 1))
  color = cp.zeros((ele, ele, height, width))
  p_c = cp.zeros((height+2*pad, width+2*pad))
  p_c[pad:height+pad, pad:width+pad] = c
  for i in range(ele):
    kernel[i] = cp.reshape(k[i], (ele, 1, 1))
    color[:, i] = p_c[i//h_k:i//h_k+height, i%w_k:i%w_k+width]
  # print(color)
  res = color * kernel
  # print(res)
  res = cp.transpose(res, (1, 0, 2, 3))
  result = sum(res)
  return result

#Intensity distance tensor
def calculate_intensity_distance_tensor(size):
  center = size // 2
  element = size * size
  intensity_distance_tensor = cp.zeros([element, size, size])
  intensity_distance_tensor[:, center, center] = -1
  intensity_distance_tensor[element//2, center, center] = 0
  for i in range(element // 2):
    intensity_distance_tensor[i, i // size, i % size] += 1
    intensity_distance_tensor[element - 1 - i, size - 1 - i // size, size - 1 - i % size] += 1

  return intensity_distance_tensor

#Space kernel
def calculate_gaussian_space(size, s_sigma):
  # space_distance_kernel = calculate_distance_kernel(size)
  space_distance_kernel = calculate_norm_distance_kernel(size)
  return cp.reshape(gaussian_function(space_distance_kernel, s_sigma), (size*size, 1, 1))

#Calculate gaussian intensity
def calculate_gaussian_intensity(c, i_sigma, intensity_distance_tensor):
  c_w, c_h = c.shape
  ele = len(intensity_distance_tensor)
  intensity_tensor = cp.zeros([ele, c_w, c_h])
  i_kernel = cp.abs(n_conv(c, intensity_distance_tensor))
  intensity_tensor = gaussian_function(i_kernel, i_sigma)

  return intensity_tensor

#Bilateral filter
def bilateral_filter(img, k_size, sigma_s, sigma_i):
  img = cp.asarray(img / 255.0)
  ele = k_size * k_size
  p_size = (k_size - 1) // 2
  width, height, channel = img.shape
  padding = cp.pad(img, ((p_size, p_size), (p_size, p_size), (0, 0)), mode='constant', constant_values=0)

  padding = cp.transpose(padding, (2, 0, 1))
  temp = cp.zeros((channel, ele, width, height))
  result = cp.zeros((channel, width + p_size * 2, height + p_size * 2))
  space_kernel = calculate_gaussian_space(k_size, sigma_s)

  weight = cp.zeros((width, height, channel))

  intensity_distance_tensor = calculate_intensity_distance_tensor(k_size)

  blue = img[:, :, 0]
  green = img[:, :, 1]
  red = img[:, :, 2]

  temp[0] = calculate_gaussian_intensity(blue, sigma_i, intensity_distance_tensor) * space_kernel
  weight[:, :, 0] = sum(temp[0])
  temp[1] = calculate_gaussian_intensity(green, sigma_i, intensity_distance_tensor) * space_kernel
  weight[:, :, 1] = sum(temp[1])
  temp[2] = calculate_gaussian_intensity(red, sigma_i, intensity_distance_tensor) * space_kernel
  weight[:, :, 2] = sum(temp[2])

  for j in range(ele):
    result[:, j // k_size:width + (j // k_size), j % k_size:height + (j % k_size)] += temp[:, j] * padding[:, j // k_size:width + (j // k_size), j % k_size:height + (j % k_size)]

  result = cp.transpose(result, (1, 2, 0))

  result = result[p_size:width + p_size, p_size:height + p_size] / weight

  return cp.asnumpy(result * 255.0)

#Frames to video
def frames_to_video(frames, path_out, fps, size):
  video_out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX') , fps, size)
  for i in range(len(frames)):
    # writing to a image array
    video_out.write(frames[i])
  video_out.release()

# Video
def bilateral_filter_for_video(path_in, path_out, kernel_size=3, sigma_s=2.5, sigma_i=0.5):
  original_video = cv2.VideoCapture(path_in)
  success = 1
  cap = original_video
  ret, frame = cap.read()
  height, width, channel = frame.shape
  size = (width, height)
  (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
  if int(major_ver) < 3 :
    fps = original_video.get(cv2.cv.CV_CAP_PROP_FPS)
  else :
    fps = original_video.get(cv2.CAP_PROP_FPS)

  bf_frames = []

  print("Processing...")
  while original_video.isOpened():
    success, image = original_video.read()
    if success != 1:
      break

    bf_img = bilateral_filter(image, kernel_size, sigma_s, sigma_i)
    bf_frames.append(bf_img.astype(np.uint8))

  frames_to_video(bf_frames, path_out, fps, size)
  print("Successfully executed. Output is at " + path_out)

# Image
def bilateral_filter_for_image(path_in, path_out, kernel_size=3, sigma_s=2.5, sigma_i=0.5):
  img = cv2.imread(path_in)
  print("Processing...")
  bf_img = bilateral_filter(img, kernel_size, sigma_s, sigma_i)
  cv2.imwrite(path_out, bf_img)
  print("Successfully executed. Output is at " + path_out)

if __name__ == '__main__':
  bilateral_filter_for_image("D:\\Semester 8\\Thesis\\Draft\\0004.png", "D:\\Semester 8\\Thesis\\Draft\\bf_0004.png")
  # bilateral_filter_for_video("D:\Semester 8\Thesis\Draft\VVK_Trim.mp4", "D:\Semester 8\Thesis\Draft\out_VVK_Trim.mp4")