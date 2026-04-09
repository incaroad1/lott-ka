from src.importers.otos_importer import OtosImporter
from src.importers.hatos_importer import HatosImporter
from src.importers.skandi_importer import SkandiImporter


def get_importer(jatek: str):
    mapping = {
        "otos": OtosImporter(),
        "hatos": HatosImporter(),
        "skandi": SkandiImporter(),
    }
    if jatek not in mapping:
        raise ValueError(f"Ismeretlen játék: {jatek}")
    return mapping[jatek]
