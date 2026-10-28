import hashlib
import json
from pathlib import Path
from gpt_vln.files import write


class CampaignHold(RuntimeError):
    pass


class BudgetHold(CampaignHold):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def immutable(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Saved run identity differs: ' + str(path))
    write(path, value)
