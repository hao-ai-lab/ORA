"""Reconstruct an NVIDIA SDK wheel using authenticated remote RECORD hashes.

Intact local payload avoids downloading gigabytes again. Missing or changed
members are fetched from the exact official archive using HTTP ranges. The
donor is read-only; the result is explicitly a derived archive with a new hash.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


class RemoteArchive(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.length = int(response.headers["Content-Length"])
        self.position = 0
        self.bytes_received = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if size == 0:
            return b""
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.position}-{self.position + size - 1}",
        })
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError("Official archive did not honor bounded HTTP Range")
            data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError(f"Truncated HTTP Range: {len(data)} != {size}")
        self.position += len(data)
        self.bytes_received += len(data)
        return data


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--donor-site-packages", type=Path, required=True)
    parser.add_argument("--package", default="isaacsim-extscache-kit")
    args = parser.parse_args()
    if not args.package.startswith("isaacsim-") or any(c not in "abcdefghijklmnopqrstuvwxyz-" for c in args.package):
        parser.error("Expected an isaacsim distribution name")
    name = args.package.replace("-", "_") + "-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
    url = f"https://pypi.nvidia.com/{args.package}/{name}"
    transport = RemoteArchive(url)
    output_dir = args.root / "wheels"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / name
    temp = output.with_suffix(".whl.partial")
    report = {"official_url": url, "donor": str(args.donor_site_packages),
              "validation": "Every member matched SHA256 in official remote RECORD",
              "original_zip_hash_verified": False, "reused_files": 0, "downloaded_files": []}
    with zipfile.ZipFile(transport) as original:
        record_name = args.package.replace("-", "_") + "-5.1.0.0.dist-info/RECORD"
        record_data = original.read(record_name)
        rows = list(csv.reader(io.StringIO(record_data.decode())))
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as derived:
            for index, (relative, expected, size) in enumerate(rows):
                if relative == record_name:
                    continue
                path = args.donor_site_packages / relative
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise RuntimeError(f"Unexpected wheel member {relative}")
                if not expected.startswith("sha256="):
                    raise RuntimeError(f"Unhashed official payload member {relative}")
                required = expected.split("=", 1)[1]
                use_local = path.is_file() and path.stat().st_size == int(size)
                if use_local:
                    use_local = base64.urlsafe_b64encode(digest(path)).decode().rstrip("=") == required
                if use_local:
                    derived.write(path, relative)
                    report["reused_files"] += 1
                else:
                    data = original.read(relative)
                    actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                    if actual != required or len(data) != int(size):
                        raise RuntimeError(f"Official archive member hash mismatch: {relative}")
                    derived.writestr(relative, data)
                    report["downloaded_files"].append(relative)
                    print(f"Fetched official member {relative}", flush=True)
                if index % 5000 == 0:
                    print(f"Verified {index}/{len(rows)} files", flush=True)
            derived.writestr(record_name, record_data)
    temp.replace(output)
    report.update(derived_wheel=str(output), derived_wheel_sha256=digest(output).hex(),
                  network_bytes=transport.bytes_received,
                  original_record_sha256=hashlib.sha256(record_data).hexdigest())
    (args.root / f"{args.package}-rebuild-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
