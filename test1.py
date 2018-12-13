import numpy
import keras
from keras.models import load_model
from PIL import Image
import pandas as pd 
from keras.models import model_from_json


im = Image.open("AamirKhan_8.jpg")
im = im.convert('1')
im = im.resize((48,48),Image.ANTIALIAS)
im.save("image_scaled.jpg",quality=95)
X = numpy.array(im)
print(X)
X = X.reshape(1, 48, 48,1)
X = X.astype("float32")
X /= 255

print(X)

#app=flask.Flask(__name__)
#@app.route("/predict",methods=["GET","POST"])

def predict():
	with open('model_architecture.json', 'r') as f:
		model = model_from_json(f.read())
	data={}	
	# Load weights into the new model
	model.load_weights('model_weights.h5')
	##x=model.input
	y_prob=model.predict(X,verbose=0)
	data["prediction"]=str(y_prob)
	return data
# app.run()
print(predict())
