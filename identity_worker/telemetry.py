from __future__ import annotations
import json,time

def log_event(event: str, **fields):
    safe={k:('[redacted]' if any(token in k.lower() for token in ('token','secret','url')) else v) for k,v in fields.items()}
    print(json.dumps({'timestamp_ms':int(time.time()*1000),'event':event,**safe},ensure_ascii=False,separators=(',',':')),flush=True)
