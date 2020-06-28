import numpy as np
import matplotlib.pyplot as plt
import cv2

def autoCanny(img_Gray, sigma = 0.33):
	median = np.median(img_Gray)
	low = max(0,round((1-sigma)*median))
	canny = cv2.Canny(img_Gray,low,255)
	return canny

def drawHoughLines(img, minLineLength, maxLineGap):
	edges = np.zeros_like(img)
	lines = cv2.HoughLinesP(img,rho = 5,theta = np.pi/180,threshold = 50,minLineLength = minLineLength ,maxLineGap = maxLineGap)
	for i in range(len(lines)):
		x1,y1,x2,y2 = lines[i][0]
		cv2.line(edges,(x1,y1),(x2,y2),255,3)
	return edges

def Detect(value):
    # parameters for all the filters
	Bilateral_Kernel = 30
	Bilateral_SigmaSpace = 80
	Bilateral_SigmaColor = 50
	Canny_Sigma = 0.4
	Hough_MinLength = 40
	Hough_MaxGap = 30
	Dilate_Kernel = (5,5)
	Dilate_Iterations = 10
	path = 'path_to_image'
	img_Original = cv2.imread(path)
	img_RGB = np.copy(img_Original)
	img_RGB = cv2.cvtColor(img_RGB,cv2.COLOR_BGR2RGB)
	#img_Gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    # For detection of object edges using any edge detector, I found the HLS colorscale to be the most effective,
    # as all the colours can be differentiated using only the hue parameter. And using HLS also highly reduces the impact 
    # shadows have on object detection, as the image can be classified using only the hue, without the saturation and
    # light intensity
	img_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
	
    # Bilateral filter is helpful in grouping the nearpy pixels with similar hue, which increases the 
    # accuracy of the edge detector being used after this
	bilat = cv2.bilateralFilter( img_RGB,  Bilateral_Kernel,   Bilateral_SigmaColor,   Bilateral_SigmaSpace)
	bilat = cv2.bilateralFilter( bilat,  Bilateral_Kernel,   Bilateral_SigmaColor,   Bilateral_SigmaSpace)
	bilat = cv2.bilateralFilter( bilat,  Bilateral_Kernel,   Bilateral_SigmaColor,   Bilateral_SigmaSpace)
	
	bilat_Edge = autoCanny(  bilat,   Canny_Sigma  )
    
    # The dialtion after edge detection was helpful if removing the small 5-10 pixel clusters in edges
	bilat_Edge = cv2.dilate(  bilat_Edge,   Dilate_Kernel,  iterations = Dilate_Iterations)
    
    # As the object to be detected was a box, only straight edges are drawn by using houghlines
	bilat_Edge = drawHoughLines(  bilat_Edge  ,  Hough_MinLength,  Hough_MaxGap  )
	bilat_Edge = cv2.dilate(  bilat_Edge,   Dilate_Kernel,  iterations = Dilate_Iterations)
	#plt.subplot(2,1,1)
	#plt.imshow(bilat_Edge)
	contours, hierarchy = cv2.findContours(bilat_Edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	count = 0
	for c in contours:
		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		#print(box)
		#print(cv2.contourArea(box))
        # Just some conditions for only considering boxes of reasonable size and length and breadth being not too different,
        # as the boxes to detect were small cubes on a grassy field
		if rect[1][0] * rect[1][1]>100 and rect[1][0]/rect[1][1]<= 1.3 and rect[1][1]/rect[1][0] <= 1.3:
			#print(rect)
			M = cv2.getPerspectiveTransform(np.float32(box), np.array([[0,300],[300,300],[300,0],[0,0]],dtype = 'float32'))
			warped = cv2.warpPerspective(img_HLS, M, (300, 300))
			#print(np.sqrt(np.var(warped[50:230,50:230],axis = (0,1))))
			if np.mean(warped,axis = (0,1))[0]<35 and np.mean(warped,axis = (0,1))[0]<77 and (np.sqrt(np.var(warped,axis = (0,1))[0])<2 or np.sqrt(np.var(warped[50:230,50:230],axis = (0,1))[0])<1):
				count += 1
				img_RGB = cv2.drawContours(img_RGB,[box],0,(0,0,0),4)
	print(count)
	#plt.subplot(2,1,2)
	#plt.imshow(img_RGB)
	#plt.colorbar()
	#plt.show()
	return count

count = Detect(1)
