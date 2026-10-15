"""Check pinned Hub metadata using HEAD requests, without transferring model weights."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def inspect(item):
    url = f"https://huggingface.co/{item['repository']}/resolve/{item['revision']}/{item['file']}"
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "h3-quality-metadata-preflight/1.0"})
    opener = urllib.request.build_opener(NoRedirect)
    try:
        response = opener.open(request, timeout=45)
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f"Metadata HTTP {exc.code}: {item['file']}") from None
        response = exc
    with response:
        headers = response.headers
        size = int(headers.get("x-linked-size", headers.get("Content-Length", "0")))
        sha = headers.get("x-linked-etag", headers.get("ETag", "")).strip('"')
        if size != item["size_bytes"] or sha != item["sha256_published"]:
            raise RuntimeError(f"Pinned metadata differs for {item['file']}: size={size}, sha={sha}")
    return {"file": item["file"], "size_bytes": size, "sha256_published": sha,
            "pinned_metadata_verified": True, "payload_downloaded": False}


if __name__ == "__main__":
    items = json.loads((ROOT / "models.lock.json").read_text())["models"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(inspect, items))
    print(json.dumps({"models": result, "total_size_bytes": sum(x["size_bytes"] for x in result)}, indent=2))
