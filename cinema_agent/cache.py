import base64
import hashlib
import json
from collections import OrderedDict
from threading import RLock


def _image_bytes(image_data_url: str) -> bytes:
    if "," in image_data_url:
        _, encoded = image_data_url.split(",", 1)
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            pass
    return image_data_url.encode("utf-8")


def analysis_cache_key(image_data_url: str, screenplay: str, intent: str) -> str:
    image_hash = hashlib.sha256(_image_bytes(image_data_url)).hexdigest()
    payload = json.dumps(
        {"image_hash": image_hash, "screenplay": screenplay, "intent": intent},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AnalysisCache:
    """Small process-local cache that stores results, never uploaded image bytes."""

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = RLock()

    def get(self, key):
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def set(self, key, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)