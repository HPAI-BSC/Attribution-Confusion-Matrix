import os
import hashlib

from consts.consts import MosaicArgs, XmethodArgs, ArchArgs

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir,) * 2))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
CKPTS_PATH = os.path.join(PROJECT_PATH, 'checkpoints')


class Paths:
    formatted_pt_path = os.path.join(CKPTS_PATH, 'formatted_pt_ckpts')
    mosaics_path = os.path.join(DATA_PATH, 'mosaics')
    explainability_path = os.path.join(DATA_PATH, 'explainability')
    explainability_csv = os.path.join(explainability_path, "hash_explainability.csv")


class MosaicPaths:
    @classmethod
    def get_from(cls, dataset: MosaicArgs):
        dataset_map = {
            MosaicArgs.CATSDOGS_MOSAIC: cls.CatsDogsMosaic,
            MosaicArgs.MAME_MOSAIC: cls.MameMosaic,
            MosaicArgs.MIT67_MOSAIC: cls.Mit67Mosaic,
        }
        return dataset_map[dataset]

    class CatsDogsMosaic:
        images_folder = os.path.join(Paths.mosaics_path, 'catsdogs_mosaic', 'data')
        csv_path = os.path.join(Paths.mosaics_path, 'catsdogs_mosaic', 'catsdogs.csv')

    class Mit67Mosaic:
        images_folder = os.path.join(Paths.mosaics_path, 'mit670_mosaic', 'data')
        csv_path = os.path.join(Paths.mosaics_path, 'mit670_mosaic', 'mit670.csv')

    class MameMosaic:
        images_folder = os.path.join(Paths.mosaics_path, 'mame_mosaic', 'data')
        csv_path = os.path.join(Paths.mosaics_path, 'mame_mosaic', 'mame.csv')

def get_heatmaps_folder(xmethod: XmethodArgs, dataset: MosaicArgs, architecture: ArchArgs, ckpt: str):
    string2hash = f"{xmethod}{dataset}{architecture}{ckpt}"
    _hash = hashlib.md5(string2hash.encode('utf-8')).hexdigest()
    path = os.path.join(Paths.explainability_path, f"{_hash}")
    return _hash, path