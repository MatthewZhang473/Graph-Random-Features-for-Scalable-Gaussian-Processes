# Wind interpolation datasets


**Step 1**: Register your Climate Data Store (CDS) account: https://cds.climate.copernicus.eu/how-to-api


**Step 2**: Use the following python script to acquire the wind_interpolation dataset:


```
import cdsapi

dataset = "reanalysis-era5-pressure-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "pressure_level": ["500", "800", "1000"],
    "year": ["2010"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
```