import numpy as np
from dataclasses import dataclass
import datetime


@dataclass
class satellite_amf:
    vcd: np.ndarray
    scd: np.ndarray
    time: datetime.datetime
    tropopause: np.ndarray
    latitude_center: np.ndarray
    longitude_center: np.ndarray
    latitude_corner: np.ndarray
    longitude_corner: np.ndarray
    uncertainty: np.ndarray
    quality_flag: np.ndarray
    pressure_mid: np.ndarray
    scattering_weights: np.ndarray
    ctm_upscaled_needed: bool
    ctm_vcd: np.ndarray
    ctm_time_at_sat: datetime.datetime
    surface_albedo: np.ndarray
    SZA: np.ndarray
    surface_alt: np.ndarray


@dataclass
class ctm_model:
    latitude: np.ndarray
    longitude: np.ndarray
    time: list
    gas_profile_no2: np.ndarray
    gas_profile_hcho: np.ndarray
    O3col: np.ndarray
    pressure_mid: np.ndarray
    tempeature_mid: np.ndarray
    height_mid: np.ndarray
    PBLH: np.ndarray
    ctmtype: str


@dataclass
class param_input:
    longitude: np.ndarray
    latitude: np.ndarray
    time: datetime.datetime
    gas_profile_no2: np.ndarray
    gas_profile_hcho: np.ndarray
    O3col: np.ndarray
    h2o: np.ndarray
    pressure_mid: np.ndarray
    tempeature_mid: np.ndarray
    height_mid: np.ndarray
    PBLH: np.ndarray
    vcd: np.ndarray
    vcd_err: np.ndarray
    tropopause: np.ndarray
    surface_albedo: np.ndarray
    SZA: np.ndarray
    surface_alt: np.ndarray

@dataclass
class param_output:
    latitude: np.ndarray
    longitude: np.ndarray
    vcd_no2: np.ndarray
    vcd_no2_factor: np.ndarray
    vcd_hcho: np.ndarray
    vcd_hcho_factor: np.ndarray
    PO3: np.ndarray
    FNR: np.ndarray
    hcho_vmr: np.ndarray
    no2_vmr: np.ndarray
    jno2: np.ndarray
    jo1d: np.ndarray
    hcho_vmr_contrib: np.ndarray
    no2_vmr_contrib: np.ndarray
    jno2_contrib: np.ndarray
    jo1d_contrib: np.ndarray
    po3_err: np.ndarray
      
