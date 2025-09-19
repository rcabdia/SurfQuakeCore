from dataclasses import dataclass, asdict
from surfquakecore.utils import Cast


@dataclass
class BaseDataClass:

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dto: dict):
        return Cast(dto, cls)
