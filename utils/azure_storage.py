import os

from azure.storage.blob import BlobServiceClient


def download_file_from_azure(container_name, blob_name):

    blob_service_client = BlobServiceClient.from_connection_string(os.environ.get('AZURE_STORAGE_CONNECTION_STRING'))
    container_client = blob_service_client.get_container_client(container=container_name)

    with open(file=blob_name, mode="wb") as download_file:
        download_file.write(container_client.download_blob(blob_name).readall())
