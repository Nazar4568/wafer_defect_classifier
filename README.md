# Wafer Defect Inspection API

## Description
This project is an end-to-end Machine Learning microservice built to classify surface defects on semiconductor wafers. 
Initially developed as a pure ML training pipeline, it has been expanded into a fully functional REST API. The system processes incoming electron microscope images through a custom PyTorch Convolutional Neural Network (CNN), saves the images locally, and logs all prediction results (including confidence scores and timestamps) into a PostgreSQL database.

The entire system, including the API and the database, is containerized for seamless deployment.

## Tech Stack
* **Backend:** FastAPI, Pydantic, Python 3
* **Database:** PostgreSQL, SQLAlchemy 2.0 (ORM), psycopg2
* **Machine Learning:** PyTorch, Torchvision
* **Deployment:** Docker, Docker Compose
* **Metrics & Evaluation:** Scikit-learn, Matplotlib, Seaborn

## Features
* **Real-time Inference:** REST API endpoint that accepts image uploads and returns defect classifications instantly.
* **Persistent Storage:** Images are securely saved with UUIDs, and all inspection metadata is logged into a relational database.
* **Interactive Documentation:** Out-of-the-box Swagger UI for easy API testing and exploration.

## What it detects
The CNN classifies images into 6 types of anomalies based on the NEU Surface Defect Database:
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, and `scratches`.

## Model Results
Tested on a validation set of 360 images (60 per class).
* **Overall Accuracy:** 84%

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Crazing | 1.00 | 0.85 | 0.92 |
| Inclusion | 0.91 | 0.70 | 0.79 |
| Patches | 0.72 | 1.00 | 0.84 |
| Pitted Surface | 0.81 | 0.92 | 0.86 |
| Rolled-in Scale | 0.93 | 0.70 | 0.80 |
| Scratches | 0.78 | 0.87 | 0.82 |

## How to run it

**1. Clone the repository**
`bash
git clone https://github.com/Nazar4568/wafer_defect_classifier.git
`

`bash
cd wafer_defect_classifier
`

**2. Start the Microservice (API + PostgreSQL)**
Make sure you have Docker Desktop installed and running. Run the following command in the project root:
`bash
docker-compose up -d --build
`
*This command will automatically download PostgreSQL, install all dependencies, build the API image, and start both containers. The database tables are generated automatically.*

**3. Test the API**
Once the containers are running, navigate to:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

From the Swagger UI:
1. Click on the `POST /api/v1/predict` route.
2. Click **Try it out**.
3. Upload any defect image and hit **Execute**.

**4. Stop the Microservice**
To safely shut down the containers without losing database records:
`bash
docker-compose down
`
