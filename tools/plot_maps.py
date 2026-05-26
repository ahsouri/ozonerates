import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
# -------------------------
# Settings
# -------------------------
data_dir = "/media/asouri/Amir_5TB1/NASA/PO3_Product/TROPOMI/data/netcdf/"
file_pattern = "PO3_NW__TROPOMI_____{}_*.nc"
lon_min, lon_max = -86, -80
lat_min, lat_max = 30, 35
# If days are omitted or set to None -> use all days
day_selection = {
    "202308": None,   # all days
    "202309": [10,12]
}

def to_numpy(x):
    return np.ma.filled(x[:], np.nan).astype("float32")


def read_vars(fname, selected_days=None):
    with Dataset(fname, "r") as nc:
        time = nc["time"][:]

        dates = num2date(
            time,
            units=nc["time"].units,
            calendar=getattr(nc["time"], "calendar", "gregorian"),
            only_use_python_datetimes=True
        )
        # Extract day-of-month
        days = np.array([d.day for d in dates])

        # Select indices
        if selected_days is None:
            keep = np.ones(len(days), dtype=bool)
        else:
            keep = np.isin(days, selected_days)

        PO3 = np.ma.filled(nc["PO3"][keep, ...], np.nan)
        PO3_NO2 = np.ma.filled(nc["PO3_NO2"][keep, ...], np.nan)
        PO3_HCHO = np.ma.filled(nc["PO3_HCHO"][keep, ...], np.nan)
        NO2 = np.ma.filled(nc["NO2_ppbv"][keep, ...], np.nan)
        HCHO = np.ma.filled(nc["HCHO_ppbv"][keep, ...], np.nan)
        jNO2 = np.ma.filled(nc["JNO2"][keep, ...], np.nan)

        lat = np.ma.filled(nc["latitude"][:], np.nan)
        lon = np.ma.filled(nc["longitude"][:], np.nan)

    return PO3, PO3_NO2, PO3_HCHO, NO2, HCHO, jNO2, lat, lon

# -------------------------
# Load data
# -------------------------


data = {
    "PO3": [],
    "PO3_NO2": [],
    "PO3_HCHO": [],
    "NO2": [],
    "HCHO": [],
    "jNO2": [],
    "lat": [],
    "lon": []
}

for m, selected_days in day_selection.items():
    selected_days = day_selection.get(m, None)
    # we focus on NW because the US
    fname_pattern = [data_dir, f"PO3_NW__TROPOMI_____{m}_*.nc"]
    fname_pattern = "".join(fname_pattern)
    fname = glob.glob(fname_pattern)
    if len(fname) == 0:
        print(f"No file found for {m}")
        continue
    print(fname)

    PO3, PO3_NO2, PO3_HCHO, NO2, HCHO, jNO2, lat, lon = read_vars(
        fname[0], selected_days=selected_days)

    data["PO3"].append(PO3)
    data["PO3_NO2"].append(PO3_NO2)
    data["PO3_HCHO"].append(PO3_HCHO)
    data["NO2"].append(NO2)
    data["HCHO"].append(HCHO)
    data["jNO2"].append(jNO2)

if len(data["PO3"]) > 1:  # multiple months

    PO3 = np.nanmean(np.concatenate(data["PO3"], axis=0), axis=0).squeeze()
    PO3_NO2 = np.nanmean(np.concatenate(
        data["PO3_NO2"], axis=0), axis=0).squeeze()
    PO3_HCHO = np.nanmean(np.concatenate(
        data["PO3_HCHO"], axis=0), axis=0).squeeze()
    NO2 = np.nanmean(np.concatenate(data["NO2"], axis=0), axis=0).squeeze()
    HCHO = np.nanmean(np.concatenate(data["HCHO"], axis=0), axis=0).squeeze()
    jNO2 = np.nanmean(np.concatenate(data["jNO2"], axis=0), axis=0).squeeze()


else:
    PO3 = np.nanmean(np.concatenate(data["PO3"], axis=0), axis=0)
    PO3_NO2 = np.nanmean(np.concatenate(data["PO3_NO2"], axis=0), axis=0)
    PO3_HCHO = np.nanmean(np.concatenate(data["PO3_HCHO"], axis=0), axis=0)
    NO2 = np.nanmean(np.concatenate(data["NO2"], axis=0), axis=0)
    HCHO = np.nanmean(np.concatenate(data["HCHO"], axis=0), axis=0)
    jNO2 = np.nanmean(np.concatenate(data["jNO2"], axis=0), axis=0)

# empty the memory
data = []
# -------------------------
# plotting
# --------------------------
plot_data = {
    "PO3": PO3,
    "PO3_NO2": PO3_NO2,
    "PO3_HCHO": PO3_HCHO,
    "NO2": NO2,
    "HCHO": HCHO,
    "jNO2": jNO2,
}

vars_to_plot = ["PO3", "PO3_NO2", "PO3_HCHO", "NO2", "HCHO", "jNO2"]
titles = ["PO3 PBL", "PO3_NO2", "PO3_HCHO", "NO2 PBL", "HCHO PBL", "jNO2"]

# Make sure these are plain arrays
lon = np.asarray(lon).squeeze()
lat = np.asarray(lat).squeeze()

# Crop indices
i_keep = np.where((lat[:, 0] >= lat_min) & (lat[:, 0] <= lat_max))[0]
j_keep = np.where((lon[0, :] >= lon_min) & (lon[0, :] <= lon_max))[0]

ii = slice(i_keep.min(), i_keep.max() + 1)
jj = slice(j_keep.min(), j_keep.max() + 1)

lon_c = lon[ii, jj]
lat_c = lat[ii, jj]


fig = plt.figure(figsize=(15, 8))
proj = ccrs.PlateCarree()

for i, varname in enumerate(vars_to_plot):

    ax = fig.add_subplot(2, 3, i + 1, projection=proj)

    data_plot = np.asarray(plot_data[varname]).squeeze()

    data_c = data_plot[ii, jj].astype("float32")

    # Replace inf with nan
    data_c[~np.isfinite(data_c)] = np.nan

    if varname in ["PO3", "PO3_NO2", "PO3_HCHO"]:
        vmin, vmax = -1, 8
        cmap = "turbo"

    elif varname == "HCHO":
        vmin, vmax = 0, 5
        cmap = "rainbow"

    elif varname == "NO2":
        vmin, vmax = 0, 2
        cmap = "rainbow"

    elif varname == "jNO2":
        vmin, vmax = 0, 1.8e-2
        cmap = "plasma"

    im = ax.pcolormesh(
        lon_c, lat_c, data_c,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        shading="auto",
        transform=proj
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles[i], fontsize=16)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.04
    )
    if varname in ["PO3", "PO3_NO2", "PO3_HCHO"]:
        cbar.set_label(r"ppbv/hr", fontsize=12)
    if varname in ["HCHO", "NO2"]:
        cbar.set_label(r"ppbv", fontsize=12)
    if varname in ["jNO2"]:
        cbar.set_label(r"1/s", fontsize=12)


plt.tight_layout()
fig.savefig("PO3_sample.png", dpi=300, bbox_inches="tight")
plt.show()
