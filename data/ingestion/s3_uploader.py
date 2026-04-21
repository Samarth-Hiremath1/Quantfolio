import os
import boto3
import logging
from botocore.exceptions import ClientError
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class S3Uploader:
    """
    Handles uploading raw datasets to AWS S3 (or a LocalStack emulator in dev).
    """

    def __init__(self, bucket_name: str, endpoint_url: Optional[str] = None):
        """
        :param bucket_name: The target S3 bucket name.
        :param endpoint_url: Override URL for the S3 endpoint (useful for LocalStack, e.g., http://localhost:4566).
        """
        self.bucket_name = bucket_name
        
        # In a real setup, access keys are picked up automatically by boto3.
        # Here we mock it or pass it for LocalStack compatibility.
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url or os.getenv("LOCALSTACK_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )

        # Auto-create bucket to prevent NoSuchBucket errors on first run
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            logger.info(f"Bucket {self.bucket_name} not found. Creating it now...")
            self.s3_client.create_bucket(Bucket=self.bucket_name)

    def upload_csv(self, file_path: str, object_name: Optional[str] = None) -> bool:
        """
        Uploads a local CSV file to the S3 bucket.
        
        :param file_path: Path to the local file.
        :param object_name: Key to save the object as in S3. If None, uses the file_path basename.
        :return: True if upload successful, False otherwise.
        """
        if object_name is None:
            object_name = os.path.basename(file_path)

        logger.info(f"Uploading {file_path} to s3://{self.bucket_name}/{object_name}")
        
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            logger.info("Upload successful.")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False

    def upload_dataframe(self, df, object_name: str) -> bool:
        """
        Convenience method to upload a pandas DataFrame directly as CSV without a local staging file.
        """
        import io
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        logger.info(f"Uploading DataFrame directly to s3://{self.bucket_name}/{object_name}")
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name, 
                Key=object_name, 
                Body=csv_buffer.getvalue()
            )
            logger.info("Upload successful.")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload DataFrame to S3: {e}")
            return False
