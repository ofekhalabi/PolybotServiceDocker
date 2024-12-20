import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
from pymongo import MongoClient
import yaml
import boto3
from botocore.exceptions import NoCredentialsError
from loguru import logger
import os


images_bucket = os.environ['BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # TODO download img_name from S3, store the local image path in the original_img_path variable.
    #  The bucket name is provided as an env var BUCKET_NAME.
    # Initialize the S3 client
    s3 = boto3.client('s3')
    original_img_path = f'/tmp/{img_name}'  # Temporary storage for downloaded image
    try:
        # Download the file from S3
        s3.download_file(images_bucket, img_name, original_img_path)
        logger.info(f'prediction: {prediction_id}. Downloaded {img_name} to {original_img_path}')
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    except Exception as e:
        logger.error(f'Error downloading {img_name}: {e}')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = f'static/data/{prediction_id}/{img_name}'

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    # Specify the local file and S3 bucket details
    s3_image_key_upload = f'predictions/{prediction_id}_{img_name}'

    try:
        # Upload predicted image back to S3
        s3.upload_file(str(predicted_img_path), images_bucket, s3_image_key_upload)
        logger.info(f"File uploaded successfully to {images_bucket}/{s3_image_key_upload}")
    except FileNotFoundError:
        logger.error("The file was not found.")
        return "Predicted image not found", 404
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
        return "AWS credentials not available", 403
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return f"Error uploading file: {e}", 500

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{Path(original_img_path).stem}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB
        # Connect to MongoDB
        client = MongoClient('mongodb://MongoDBPrimary:27017,MongoDBSec1:27018,MongoDBSec2:27019/?replicaSet=myReplicaSet')


        # Select the database and collection
        db = client['polybot-info']
        collection = db['prediction_images']
        # Insert the prediction_summary into MongoDB
        collection.insert_one(prediction_summary)
        print("Prediction summary inserted successfully.")
        if "_id" in prediction_summary:
            prediction_summary["_id"] = str(prediction_summary["_id"])

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
