import os
from typing import IO


def download_file(s3_client, bucket: str, key: str, local_path: str):
    """Download an S3 object to a local file."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)


def upload_file(s3_client, bucket: str, key: str, local_path: str):
    """Upload a local file to S3."""
    s3_client.upload_file(local_path, bucket, key)


def upload_fileobj(s3_client, bucket: str, key: str, fileobj: IO):
    """Upload a file-like object to S3."""
    s3_client.upload_fileobj(fileobj, bucket, key)


def list_keys(s3_client, bucket: str, prefix: str) -> list[str]:
    """List all object keys under a prefix (recursive)."""
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def list_immediate_prefixes(s3_client, bucket: str, prefix: str) -> list[str]:
    """List immediate 'subdirectory' prefixes under a given prefix (non-recursive)."""
    if not prefix.endswith("/"):
        prefix += "/"
    prefixes = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])
    return prefixes


def list_zip_keys(s3_client, bucket: str, prefix: str = "") -> list[str]:
    """List all .zip object keys under a prefix."""
    return [k for k in list_keys(s3_client, bucket, prefix) if k.endswith(".zip")]
