from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from PIL import Image
import os
import tempfile
from utils.model import Predict_FaceModel

from fastapi.templating import Jinja2Templates

# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware

template = Jinja2Templates(directory="template")

SAVE_DIR = "outputs_app/"
# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load pretrained model
model = Predict_FaceModel.load_model('models/LPIPS_loss.h5')

# Create FastAPI app
app = FastAPI()

'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to allow requests from your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
'''
@app.get("/")
async def read_root(request: Request):
    return template.TemplateResponse("base_template.html", {'request': request})

# Define prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    
    # Save image to a temporary file
    temp_img_fd, temp_img_path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(temp_img_fd, "wb") as temp_img_file:
        temp_img_file.write(contents)
    
    # Perform prediction
    generated_output = model.predict(temp_img_path)
    
    # Save the generated image to local filesystem
    generated_img_path = os.path.join(SAVE_DIR, "generated_output.png")
    generated_output.save(generated_img_path)
    
     # Remove the temporary image file
    os.remove(temp_img_path)
    
    # Check if the saved image exists
    if not os.path.exists(generated_img_path):
        raise HTTPException(status_code=404, detail="Generated image not found")
    
    # Return the saved image file
    return FileResponse(generated_img_path, media_type="image/png", headers={"Content-Disposition": "inline; filename=generated_output.png"})