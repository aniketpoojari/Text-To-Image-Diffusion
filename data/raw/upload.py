import boto3
s3Resource = boto3.resource('s3')

try:
    s3Resource.meta.client.upload_file(
        'flowers.zip', 
        'text-to-image-aniket', 
        'flowers.zip')

except Exception as exp:
    print('exp: ', exp)