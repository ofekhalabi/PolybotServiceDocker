import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
from polybot.img_proc import Img
import boto3
from botocore.exceptions import NoCredentialsError
import requests


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ImageProcessingBot(Bot):
    def handle_message(self, msg):
        try:
            logger.info(f'Incoming message: {msg}')
            chat_id = msg['chat']['id']
            caption = msg.get('caption', None)  # Get the caption from the message
            if self.is_current_msg_photo(msg):
                photo_path = self.download_user_photo(msg)

                if caption:
                    caption = caption.lower()
                    # Check if the caption matches any of the defined commands
                    if caption in ['blur', 'contour', 'rotate', 'segment', 'salt and pepper', 'concat', 'rotate 2']:
                        processed_image_path = self.process_image(photo_path, caption)
                        self.send_photo(chat_id,processed_image_path)

                    else:
                        self.send_text(chat_id, f'Unknown command: {caption}')
                else:
                    self.send_text(chat_id, 'Photo received with no caption.')
            else:
                self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self.send_text(msg['chat']['id'], "There was an error processing the image.")


    def process_image(self, photo_path, command):
        try:
            img = Img(photo_path)  # Assuming Img is your image processing class
            if command == 'blur':
                img.blur()
            elif command == 'contour':
                img.contour()
            elif command == 'rotate':
                img.rotate()
            elif command == 'segment':
                img.segment()
            elif command == 'salt and pepper':
                img.salt_n_pepper()
            elif command == 'concat':
                img.concat(img)
            elif command == 'rotate 2':
                img.rotate()
                img.rotate()
            return img.save_img()
        except Exception as e:
            logger.error(f'Error processing image: {e}')

class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)

            # TODO upload the photo to S3
            # upload the image to S3 Bucket ofekh-polybotservicedocker-project
            s3 = boto3.client('s3')
            bucket_name = os.getenv('BUCKET_NAME')
            s3_image_key_upload = f'telegramBOT/picture.jpg'

            try:
                # Upload predicted image back to S3
                s3.upload_file(str(photo_path), bucket_name, s3_image_key_upload)
                logger.info(f"File uploaded successfully to {bucket_name}/{s3_image_key_upload}")
            except FileNotFoundError:
                logger.error("The file was not found.")
                return "Predicted image not found", 404
            except NoCredentialsError:
                logger.error("AWS credentials not available.")
                return "AWS credentials not available", 403
            except Exception as e:
                logger.error(f"Error uploading file: {e}")
                return f"Error uploading file: {e}", 500
            # TODO send an HTTP request to the `yolo5` service for prediction

            url = "http://localhost:8081/predict"
            params = {"imgName": s3_image_key_upload}

            response = requests.post(url, params=params)
            # Print the response from the server
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)
            # TODO send the returned results to the Telegram end-user
