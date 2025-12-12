import pandas as pd
import numpy as np
import folium
import h3
import branca.colormap as cm
from huggingface_hub import hf_hub_download
import lightgbm as lgb
from shapely.geometry import shape, Point

# Morocco Boundary (High precision)

morocco_boundary = [
    [-5.193863, 35.755182], [-4.591006, 35.330712], [-3.640057, 35.399855],
    [-2.604306, 35.179093], [-2.169914, 35.168396], [-1.792986, 34.527919],
    [-1.733455, 33.919713], [-1.388049, 32.864015], [-1.124551, 32.651522],
    [-1.307899, 32.262889], [-2.616605, 32.094346], [-3.06898, 31.724498],
    [-3.647498, 31.637294], [-3.690441, 30.896952], [-4.859646, 30.501188],
    [-5.242129, 30.000443], [-6.060632, 29.7317], [-7.059228, 29.579228],
    [-8.674116, 28.841289], [-8.66559, 27.656426], [-8.817809, 27.656426],
    [-8.817828, 27.656426], [-8.794884, 27.120696], [-9.413037, 27.088476],
    [-9.735343, 26.860945], [-10.189424, 26.860945], [-10.551263, 26.990808],
    [-11.392555, 26.883424], [-11.71822, 26.104092], [-12.030759, 26.030866],
    [-12.500963, 24.770116], [-13.89111, 23.691009], [-14.221168, 22.310163],
    [-14.630833, 21.86094], [-14.750955, 21.5006], [-17.002962, 21.420734],
    [-17.020428, 21.42231], [-16.973248, 21.885745], [-16.589137, 22.158234],
    [-16.261922, 22.67934], [-16.326414, 23.017768], [-15.982611, 23.723358],
    [-15.426004, 24.359134], [-15.089332, 24.520261], [-14.824645, 25.103533],
    [-14.800926, 25.636265], [-14.43994, 26.254418], [-13.773805, 26.618892],
    [-13.139942, 27.640148], [-13.121613, 27.654148], [-12.618837, 28.038186],
    [-11.688919, 28.148644], [-10.900957, 28.832142], [-10.399592, 29.098586],
    [-9.564811, 29.933574], [-9.814718, 31.177736], [-9.434793, 32.038096],
    [-9.300693, 32.564679], [-8.657476, 33.240245], [-7.654178, 33.697065],
    [-6.912544, 34.110476], [-6.244342, 35.145865], [-5.929994, 35.759988],
    [-5.193863, 35.755182]
]



def load_model():
    model_path = hf_hub_download(
        repo_id="AchG/Fire_Foresight",
        filename="lightgbm_fire_model.txt"
    )
    return lgb.Booster(model_file=model_path)



def load_morocco_polygon():
    morocco_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Morocco"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [morocco_boundary]
                }
            }
        ]
    }
    return shape(morocco_geojson["features"][0]["geometry"]), morocco_geojson



def filter_points(df, polygon):
    return df[df.apply(lambda r: polygon.contains(
        Point(r["longitude"], r["latitude"])
    ), axis=1)]



def generate_h3_grid(morocco_geojson, df, resolution=6):
    morocco_h3 = h3.polyfill(
        morocco_geojson["features"][0]["geometry"],
        resolution,
        geo_json_conformant=True
    )

    df["cell_id"] = df.apply(
        lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], resolution),
        axis=1
    )

    df = df[df["cell_id"].isin(morocco_h3)]

    h3_df = (
        df.groupby("cell_id")
                .agg(fire_prob=("predicted_probability", "mean"))
                .reset_index()
    )

    h3_df["boundary"] = h3_df["cell_id"].apply(
        lambda c: {"type": "Polygon", "coordinates": [h3.h3_to_geo_boundary(c, True)]}
    )
    h3_df["center"] = h3_df["cell_id"].apply(h3.h3_to_geo)

    return h3_df



def build_map(h3_df, output_path="morocco_fire_map.html"):
    m = folium.Map(location=[31.5, -7.0], zoom_start=6)

    min_p = h3_df["fire_prob"].min()
    max_p = h3_df["fire_prob"].max()

    colormap = cm.LinearColormap(
        ["#00FFFF", "#F1C40F", "#E74C3C"],
        vmin=min_p, vmax=max_p
    ).add_to(m)

    for _, row in h3_df.iterrows():
        tooltip = folium.GeoJsonTooltip(
            fields=["fire_prob", "cell_id", "lat", "lon"],
            aliases=[
                "Fire Probability:",
                "H3 Cell:",
                "Latitude:",
                "Longitude:"
            ],
            localize=True
        )

        folium.GeoJson(
            {
                "type": "Feature",
                "geometry": row["boundary"],
                "properties": {
                    "fire_prob": float(row["fire_prob"]),
                    "cell_id": row["cell_id"],
                    "lat": row["center"][0],
                    "lon": row["center"][1]
                },
            },
            style_function=lambda f: {
                "fillColor": colormap(f["properties"]["fire_prob"]),
                "color": "black",
                "weight": 0.3,
                "fillOpacity": 0.7,
            },
            tooltip=tooltip
        ).add_to(m)


    m.save(output_path)
    print(f"🔥 Map saved to: {output_path}")



def run_h3_processing(
        df,
        output_map="morocco_fire_map.html",
        resolution=6   # <--- NEW
    ):
    print("🔥 Loading model...")
    model = load_model()

    print("🇲🇦 Loading Morocco polygon...")
    morocco_polygon, morocco_geojson = load_morocco_polygon()

    print("📄 Loading dataset...")
    # df = pd.read_csv(input_csv)

    print("📍 Filtering points inside Morocco...")
    df = filter_points(df, morocco_polygon)

    print("🤖 Predicting...")
    features = [
        "temperature_max","wind_speed_max","precipitation_total",
        "relative_humidity","soil_moisture","evapotranspiration",
        "shortwave_radiation","day_of_year","day_of_week","is_weekend",
        "longitude","latitude","sea_distance"
    ]
    df["predicted_probability"] = model.predict(df[features])

    print(f"🧩 Generating H3 grid at resolution {resolution} ...")
    h3_df = generate_h3_grid(morocco_geojson, df, resolution=resolution)

    print("🗺 Building interactive map...")
    build_map(h3_df, output_path=output_map)


