import geopandas as gpd
import regionmask
import xarray as xr
from regionmask.core.regions import Regions
import numpy as np


SPAIN_PROVINCES_GEOJSON_URL = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/refs/heads/main/public/data/spain-provinces.geojson"

SPAIN_COMMUNITIES_GEOJSON_URL = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/refs/heads/main/public/data/spain-communities.geojson"

# SOURCE: https://www.ine.es/daco/daco42/codmun/cod_ccaa.htm
SPAIN_COMMUNITY_NAMES = {
    1:"Andalucía",
    2:"Aragón",
    3:"Principado de Asturias",
    4:"Illes Balears",
    5:"Canarias",
    6:"Cantabria",
    7:"Castilla y León",
    8:"Castilla - La Mancha",
    9:"Cataluña",
    10:"Comunitat Valenciana",
    11:"Extremadura",
    12:"Galicia",
    13:"Comunidad de Madrid",
    14:"Región de Murcia",
    15:"Comunidad Foral de Navarra",
    16:"País Vasco",
    17:"La Rioja",
    18:"Ceuta",
    19:"Melilla",
}

# SOURCE: https://www.ine.es/daco/daco42/codmun/cod_provincia.htm
SPAIN_PROVINCE_NAMES = {
    1:"Araba/Álava",
    2:"Albacete",
    3:"Alicante/Alacant",
    4:"Almería",
    5:"Ávila",
    6:"Badajoz",
    7:"Balears, Illes",
    8:"Barcelona",
    9:"Burgos",
    10:"Cáceres",
    11:"Cádiz",
    12:"Castellón/Castelló",
    13:"Ciudad Real",
    14:"Córdoba",
    15:"Coruña, A",
    16:"Cuenca",
    17:"Girona",
    18:"Granada",
    19:"Guadalajara",
    20:"Gipuzkoa",
    21:"Huelva",
    22:"Huesca",
    23:"Jaén",
    24:"León",
    25:"Lleida",
    26:"Rioja, La",
    27:"Lugo",
    28:"Madrid",
    29:"Málaga",
    30:"Murcia",
    31:"Navarra",
    32:"Ourense",
    33:"Asturias",
    34:"Palencia",
    35:"Palmas, Las",
    36:"Pontevedra",
    37:"Salamanca",
    38:"Santa Cruz de Tenerife",
    39:"Cantabria",
    40:"Segovia",
    41:"Sevilla",
    42:"Soria",
    43:"Tarragona",
    44:"Teruel",
    45:"Toledo",
    46:"Valencia/València",
    47:"Valladolid",
    48:"Bizkaia",
    49:"Zamora",
    50:"Zaragoza",
    51:"Ceuta",
    52:"Melilla",
}


def load_spain_communities(*args, **kwargs) -> Regions:
    # read provinces online
    gdf = gpd.read_file(SPAIN_COMMUNITIES_GEOJSON_URL)

    mask = regionmask.Regions(
        outlines=gdf.geometry,
        names=list(map(lambda x: SPAIN_COMMUNITY_NAMES[int(x)], gdf.cod_ccaa)),
        numbers=list(map(int, gdf.cod_ccaa)),
        name="Spanish Communities",
        **kwargs
    )
    return mask


def add_spain_communities_mask(ds: xr.Dataset) -> xr.Dataset:
    mask = load_spain_communities()
    mask = mask.mask_3D(ds, ).rename("mask_communities").astype(np.uint8)
    mask = mask.drop_vars(["abbrevs"])
    mask = mask.rename({"region": "community_id", "names": "community_name"})
    mask["community_id"] = mask.community_id.astype(np.uint8)
    mask["community_id"].attrs["long_name"] = "Spanish Communities ID"
    mask["community_name"].attrs["long_name"] = "Spanish Communities Names"
    ds["mask_communities"] = mask
    return ds


def load_spain_provinces(*args, **kwargs) -> Regions:
    # read provinces online
    gdf = gpd.read_file(SPAIN_PROVINCES_GEOJSON_URL)

    mask = regionmask.Regions(
        outlines=gdf.geometry,
        names=list(map(lambda x: SPAIN_PROVINCE_NAMES[int(x)], gdf.cod_prov)),
        numbers=list(map(int, gdf.cod_prov)),
        name="Spanish Provinces",
        **kwargs
    )
    return mask


def add_spain_provinces_mask(ds: xr.Dataset) -> xr.Dataset:
    mask = load_spain_provinces()
    mask = mask.mask_3D(ds, ).rename("mask_provinces").astype(np.uint8)
    mask = mask.drop_vars(["abbrevs"])
    mask = mask.rename({"region": "province_id", "names": "province_name"})
    mask["province_id"] = mask.province_id.astype(np.uint8)
    mask["province_id"].attrs["long_name"] = "Spanish Provinces ID"
    mask["province_name"].attrs["long_name"] = "Spanish Province Names"
    ds["mask_provinces"] = mask
    return ds