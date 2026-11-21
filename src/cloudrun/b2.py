"""Backblaze B2 through its S3 API: listing, verified directory transfer, small objects."""

from __future__ import annotations

import time
from pathlib import Path

from .creds import B2Creds


def _config():
    from botocore.config import Config
    base = dict(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=16)
    # botocore >= 1.36 computes streaming checksums by default; against B2 a retried upload then dies with
    # "UnseekableStreamError: Need to rewind the stream" (microlab lost checkpoint uploads to this).
    # Older botocore has no such option and no such behaviour.
    try:
        return Config(request_checksum_calculation="when_required", **base)
    except TypeError:
        return Config(**base)


def client(c: B2Creds):
    import boto3
    return boto3.client("s3", endpoint_url=c.endpoint, aws_access_key_id=c.key_id,
                        aws_secret_access_key=c.app_key, config=_config())


def remote_sizes(s3, bucket: str, prefix: str) -> dict[str, int]:
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            out[o["Key"]] = o["Size"]
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def get_bytes(s3, bucket: str, key: str) -> tuple[bytes, float] | None:
    """(body, last-modified epoch) or None when the key does not exist. Other errors raise."""
    from botocore.exceptions import ClientError
    try:
        o = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return o["Body"].read(), o["LastModified"].timestamp()


def put_bytes(s3, bucket: str, key: str, body: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body)


def _transfer_config():
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(multipart_threshold=128 * 1024**2, multipart_chunksize=128 * 1024**2, max_concurrency=16)


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def upload_dir(s3, local: Path, bucket: str, prefix: str, log=print) -> None:
    """Upload every file under `local`, skipping objects whose remote size already matches (resumable),
    then verify object-for-object."""
    local = Path(local)
    files = sorted(p for p in local.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"nothing to upload under {local}")
    have = remote_sizes(s3, bucket, prefix + "/")
    todo = [f for f in files if have.get(f"{prefix}/{f.relative_to(local)}") != f.stat().st_size]
    total = sum(f.stat().st_size for f in todo)
    log(f"{len(files)} files; {len(todo)} to upload ({human(total)}) -> s3://{bucket}/{prefix}/")
    sent, t0 = 0, time.time()
    for i, f in enumerate(todo, 1):
        s3.upload_file(str(f), bucket, f"{prefix}/{f.relative_to(local)}", Config=_transfer_config())
        sent += f.stat().st_size
        log(f"  [{i}/{len(todo)}] {f.relative_to(local)}  {human(sent)}/{human(total)} "
            f"{human(sent / max(time.time() - t0, 1e-3))}/s")
    verify_dir(s3, local, bucket, prefix)


def verify_dir(s3, local: Path, bucket: str, prefix: str) -> None:
    have = remote_sizes(s3, bucket, prefix + "/")
    bad = []
    for f in sorted(p for p in Path(local).rglob("*") if p.is_file()):
        k = f"{prefix}/{f.relative_to(local)}"
        if have.get(k) != f.stat().st_size:
            bad.append(f"{k}: local {f.stat().st_size} remote {have.get(k)}")
    if bad:
        raise RuntimeError("verify failed:\n  " + "\n  ".join(bad[:20]))


def download_prefix(s3, bucket: str, prefix: str, local: Path, log=print, require: bool = True) -> int:
    """Mirror s3://bucket/prefix/ into `local` (size-matched files are skipped), verify sizes. Returns bytes."""
    local = Path(local)
    have = remote_sizes(s3, bucket, prefix + "/")
    if not have and require:
        raise FileNotFoundError(f"nothing under s3://{bucket}/{prefix}/")
    total, t0 = sum(have.values()), time.time()
    for key, size in sorted(have.items()):
        dest = local / key[len(prefix) + 1:]
        if dest.exists() and dest.stat().st_size == size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        s3.download_file(bucket, key, str(tmp), Config=_transfer_config())
        if tmp.stat().st_size != size:
            raise RuntimeError(f"{key}: downloaded {tmp.stat().st_size} bytes, expected {size}")
        tmp.replace(dest)
    el = time.time() - t0
    log(f"s3://{bucket}/{prefix}/ -> {local}: {len(have)} objects, {human(total)} in {el:.0f}s")
    return total
