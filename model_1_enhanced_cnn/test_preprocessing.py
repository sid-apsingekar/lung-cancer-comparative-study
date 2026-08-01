import cv2
import matplotlib.pyplot as plt

from preprocessing import ImagePreprocessor


image_path = r"D:\projects\Lung Cancer Detection Using Hybrid Neural Network\model_1_enhanced_cnn\chest_ctscan\Data_1\train\normal\2 - Copy - Copy.png"

image = cv2.imread(image_path)

processor = ImagePreprocessor()

processed = processor.preprocess(image)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(processed)
plt.title("Processed")
plt.axis("off")

plt.show()