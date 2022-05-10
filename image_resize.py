import cv2

img = cv2.imread('/Users/saivishwasp/Downloads/resized-images/img_live2_tacklers.png',
                 cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ', img.shape)

# scale_percent = 60  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
width = 200
height = 200
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()