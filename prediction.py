
# face detection for the 5 Celebrity Faces Dataset
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

model = load_model('facenet_keras.h5')
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


if __name__ == '__main__':
	print('-------------Begin---------------')
	#Load face
	path = 'face.jpeg'
	face = extract_face(path)
	embedding = get_embedding(model, face)
	print(embedding)

	# load faces
	data = load('data.npz')
	# load face embeddings
	data = load('data-embedding.npz')
	trainX, trainy = data['arr_0'], data['arr_1']
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	# fit model
	model = SVC(kernel='linear', probability=True)
	model.fit(trainX, trainy)

	# test model on a random example from the test dataset
	random_face_pixels = face
	random_face_emb = embedding
	# prediction for the face
	samples = expand_dims(random_face_emb, axis=0)
	yhat_class = model.predict(samples)
	yhat_prob = model.predict_proba(samples)
	# get name
	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	# plot for fun
	pyplot.imshow(random_face_pixels)
	title = '%s (%.3f)' % (predict_names[0], class_probability)
	pyplot.title(title)
	pyplot.show()