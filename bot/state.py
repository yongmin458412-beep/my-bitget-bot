from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict

try:
    import numpy as np
except Exception:
    np = None

try:
    import orjson
except Exception:
    orjson = None

READ_JSON_LAST_ERROR: Dict[str, str] = {}


def _json_default(obj: Any):
    try:
        if isinstance(obj, datetime):
            return obj.isoformat()
    except Exception:
        pass
    try:
        if isinstance(obj, (set, tuple)):
            return list(obj)
    except Exception:
        pass
    try:
        if np is not None and isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return None


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = str(path) + ".tmp"
    try:
        if orjson is not None:
            with open(tmp, "wb") as f:
                opt = 0
                try:
                    opt |= orjson.OPT_SERIALIZE_NUMPY  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    opt |= orjson.OPT_NON_STR_KEYS  # type: ignore[attr-defined]
                except Exception:
                    pass
                f.write(orjson.dumps(data, default=_json_default, option=opt))
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            return


def read_json_safe(path: str, default=None):
    try:
        if orjson is not None:
            with open(path, "rb") as f:
                v = orjson.loads(f.read())
                READ_JSON_LAST_ERROR[str(path)] = ""
                return v
        with open(path, "r", encoding="utf-8") as f:
            v = json.load(f)
            READ_JSON_LAST_ERROR[str(path)] = ""
            return v
    except Exception:
        try:
            READ_JSON_LAST_ERROR[str(path)] = traceback.format_exc(limit=2)
        except Exception:
            READ_JSON_LAST_ERROR[str(path)] = "read_json_safe failed"
        return default


def safe_json_dumps(x: Any, limit: int = 2000) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        try:
            s = str(x)
        except Exception:
            s = ""
    if len(s) > int(limit):
        return s[: int(limit)] + "..."
    return s
