import torch      
import cv2      
import numpy as np 


print(f"PyTorch? {torch.__version__ is not None}")
print(f"OpenCV? {cv2.__version__ is not None}")
print(f"Numpy? {np.__version__ is not None}")

print(f"ускорение через видеокарту (CUDA): {torch.cuda.is_available()}")