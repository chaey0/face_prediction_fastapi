## Download trained model (- LPIPS)
https://drive.google.com/file/d/1qwyK1lWJYF7DEqa2Ey-93Rtk3Z5fTNgs/view?usp=sharing
```bash
├── models
│   └── LPIPS_loss.h5
```

## Run in terminal
```c
pip install -r requirements.txt
uvicorn main:app --reload
```
## Directory Structure
```bash
(Project Root)face_prediction_fastapi
│
├── outputs_app
│
├── input_images
│
├── generated_outputs
│
├── template
│   └── base_template.html
│
├── models
│   └── LPIPS_loss.h5
│
├── utils
│   ├── __init__.py
│   ├── architectures.py
│   ├── configuration.py
│   ├── model.py
│   └── face_detection.py
│
├── main.py
│
├── prediction.py
│
└── requirements.txt
```
