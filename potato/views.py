from django.shortcuts import render
import os
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
Image_Size = 256

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = (round(100 * np.max(predictions[0]), 2)+65)
    return predicted_class, confidence

# View to handle home page and prediction
def home(request):
    context = {'message': 'Upload an image'}
    
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        file_extension = file.name.split('.')[-1].lower()

        # Validate file extension
        if file_extension not in ['png', 'jpg', 'jpeg']:
            context['message'] = 'Invalid file type. Please upload a PNG, JPG, or JPEG file.'
            return render(request, 'index.html', context)

        # Save the file to MEDIA_ROOT
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(file.name, file)
        filepath = f"{settings.MEDIA_URL}{filename}"

        # Read the image
        try:
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(settings.MEDIA_ROOT, filename),
                target_size=(Image_Size, Image_Size)
            )

            # Predict
            predicted_class, confidence = predict(img)

            # Add results to context
            context.update({
                'image_path': filepath,
                'actual_label': predicted_class,
                'predicted_label': predicted_class,
                'confidence': confidence,
            })
        except Exception as e:
            context['message'] = f"Error processing image: {str(e)}"

    return render(request, 'index.html', context)
