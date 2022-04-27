import cv2

import numpy as np

src = cv2.imread('assets/dragon.png', cv2.IMREAD_UNCHANGED)

bgr = src[:,:,:3] # Channels 0..2
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Some sort of processing...

bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
alpha = src[:,:,3] # Channel 3
result = np.dstack([bgr, alpha]) # Add the alpha channel

cv2.imwrite('assets/51IgH_result.png', result)
cv2.imshow('assets/51IgH_result.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()