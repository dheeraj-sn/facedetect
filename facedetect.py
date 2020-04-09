# Import required headers
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import copy

# draw an image with detected objects
def draw_boxes(filename, result_list):
    # load the image
	data = plt.imread(filename)
	# plot the image
	plt.imshow(data)
	# get the context for drawing boxes
	ax = plt.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
        # draw the dots on eyes nose ..
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
		plt.axis('off')
    # show the plot
	plt.show()

# draw each face separately
def draw_faces(filename, result_list):
	# load the image
	data1 = plt.imread(filename)
	data = copy.deepcopy(data1)
	ax = plt.gca()
    #Setting up subplots
	faceCount = len(result_list)
	if(faceCount%4 == 0):
	    r = faceCount/4
	else:
	    r = faceCount/4 + 1
	c = 4
	# plot each face as a subplot
	for i in range(faceCount):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		#plt.subplot(1, len(result_list), i+1)
		plt.subplot(r,c,i+1)
		plt.axis('off')
		for key, value in result_list[i]['keypoints'].items():
			if key=='left_eye':
				lx=value[0]
				ly=value[1]
			if(key=='right_eye'):
				rx=value[0]
				ry=value[1]
		data[ly-2:ly+2,lx-2:lx+2,0]=255
		data[ry-2:ry+2,rx-2:rx+2,0]=255
		# plot face
		plt.imshow(data[y1:y2, x1:x2])
	# show the plot
	plt.show()

if __name__ == "__main__":
    #Create the detector(load model), using default weights
	detector = MTCNN()

	#load image from file
	filename = "a1.jpeg"
	pixels = plt.imread(filename)
	#detect faces in the image
	faces = detector.detect_faces(pixels)
	#Remove false faces
	for face in faces:
		confidence = face['confidence']
		if(confidence < 0.9):
			faces.remove(face)
	# Draw boxes on faces
	draw_boxes(filename, faces)
	# Remove individual faces
	draw_faces(filename, faces)



	#load image from file
	filename = "a2.jpeg"
	pixels = plt.imread(filename)
	#detect faces in the image
	faces = detector.detect_faces(pixels)
	#Remove false faces
	for face in faces:
		confidence = face['confidence']
		if(confidence < 0.9):
			faces.remove(face)
	# Draw boxes on faces
	draw_boxes(filename, faces)
	# Remove individual faces
	draw_faces(filename, faces)



	#load image from file
	filename = "a3.jpeg"
	pixels = plt.imread(filename)
	#detect faces in the image
	faces = detector.detect_faces(pixels)
	#Remove false faces
	for face in faces:
		confidence = face['confidence']
		if(confidence < 0.9):
			faces.remove(face)
	# Draw boxes on faces
	draw_boxes(filename, faces)
	# Remove individual faces
	draw_faces(filename, faces)
