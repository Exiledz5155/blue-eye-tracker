import cv2
import numpy as np

# Create a black image
image = np.zeros((512, 512, 3), np.uint8)

# Display the image
cv2.imshow("Test Window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
