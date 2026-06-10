import cv2, numpy as np

img = cv2.imread("./debug_base.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tune these on your lighting; green hue ~35-85 in OpenCV's 0-180 scale
mask = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

cv2.imshow("mask", mask)
cv2.waitKey(0)

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    area = cv2.contourArea(c)
    if area < 200:            # drop specks
        continue
    M = cv2.moments(c)
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]      # centroid (pixels)
    (rx, ry), (w, h), ang = cv2.minAreaRect(c)          # center + orientation
    print(f"Area: {area:.1f}  Centroid: ({cx:.1f}, {cy:.1f})  Rect center: ({rx:.1f}, {ry:.1f})  Size: ({w:.1f}, {h:.1f})  Angle: {ang:.1f}")
    cv2.circle(img, (int(cx), int(cy)), 5, (0,0,255), -1)
    box = cv2.boxPoints(((rx, ry), (w, h), ang))
    box = np.intp(box)
    cv2.drawContours(img, [box], 0, (255,0,0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)