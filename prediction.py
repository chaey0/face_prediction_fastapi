# Necessary imports
import os
from utils.model import Predict_FaceModel

# Load pretrained model
model = Predict_FaceModel.load_model('models\LPIPS_loss.h5')

# Define the path to your specific image
input_img = 'input_images/000000.png'

# Preprocess the image if necessary (resize, normalize, etc.)
# For example:
# input_img = preprocess_image(input_img)

# Perform object detection on the input image using the model
generated_output = model.predict(input_img)
output_dir = 'generated_outputs'
output_filename = 'generated_output.png'
output_path = os.path.join(output_dir, output_filename)

generated_output.save(output_path)

print(f"Generated output saved to: {output_path}")
