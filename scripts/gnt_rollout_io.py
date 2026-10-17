"""Hashing and atomic progress writes, usable without GPU dependencies."""
import hashlib
import json


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
