#!/bin/bash
set -x
# Initialize AWS resources on LocalStack startup
echo "Initializing LocalStack resources..."

# S3 bucket for raw data ingestion
awslocal s3 mb s3://quantfolio-raw-data

echo "LocalStack initialization complete."
