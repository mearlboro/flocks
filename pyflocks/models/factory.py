#!/usr/bin/python
from enum import Enum

from pyflocks.models.model import Flock, FlockModel
from pyflocks.models.vicsek import VicsekModel
from pyflocks.models.reynolds import ReynoldsModel
from pyflocks.models.kuravicsek import KuramotoVicsekModel


class EnumModels(Enum):
    UNKNOWN = 0
    VICSEK = 1
    REYNOLDS = 2
    KURAFLOCK = 3

    def __str__(self) -> str:
        return self.name.capitalize()

    @staticmethod
    def from_str(label: str) -> 'EnumModels':
        if label.upper() == 'VICSEK':
            return EnumModels.VICSEK
        elif label.upper() == 'REYNOLDS':
            return EnumModels.REYNOLDS
        elif label.upper() in ('KURAFLOCK', 'KURAVICSEK', 'KURAMOTOVICSEK'):
            return EnumModels.KURAFLOCK
        else:
            return EnumModels.UNKNOWN


class FlockFactory:
    @classmethod
    def load(self, path: str) -> FlockModel:

        mstr  = path.split('/')[-1]
        mname = mstr.split('_')[0]
        mtype = EnumModels.from_str(mname)

        if mtype != EnumModels.UNKNOWN:
            print(f"Loading {mname} model (of type {mtype}) from {path} using FlockFactory")

        if mtype == EnumModels.VICSEK:
            return VicsekModel.load(path)
        elif mtype == EnumModels.REYNOLDS:
            return ReynoldsModel.load(path)
        elif mtype == EnumModels.KURAFLOCK:
            return KuramotoVicsekModel.load(path)
        else:
            print(f"Loading experimental data from {path} using FlockFactory")
            return Flock.load(path)

