import app.perceptron as models
import app.modeling as data

#Load images and normalize them
images = data.load_images('data/images')

images = data.normalize_data(images)
print('> Data normalized!, shape: ',images.shape)

#Load model
model = models.Image_Classifier()
y_pred = model.predict(images)

print(y_pred)
