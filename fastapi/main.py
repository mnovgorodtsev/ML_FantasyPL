from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import pandas as pd
from model_class import FPLModel

app = FastAPI(title="Fantasy PL API")

df = pd.read_csv("data.csv")

fpl = FPLModel()
fpl.train(13)
fpl.train_production(current_gw=13)

@app.get("/")
def root():
    return {"message": "Fantasy PL API is running"}


@app.get("/players/{name}")
def get_player(name: str):
    result = df[df["name"].str.lower() == name.lower()]
    if result.empty:
        raise HTTPException(status_code=404, detail=f"Player not found: {name}")
    return result.to_dict(orient="records")


@app.get("/data")
def get_data(
    gw: Optional[int] = Query(None, description="Data from single X gameweek"),
    gw_max: Optional[int] = Query(None, description="Data for gameweeks 1-X")
):
    result = df.copy()

    if gw is not None:
        result = result[result["GW"] == gw]
        if result.empty:
            raise HTTPException(status_code=404, detail=f"Empty data for gameweek {gw}")

    if gw_max is not None:
        result = result[result["GW"] <= gw_max]
        if result.empty:
            raise HTTPException(status_code=404, detail=f"Empty data for gameweek {gw_max}")

    return result.to_dict(orient="records")

@app.get("/predict")
def predict(gw: int = Query(...)):
    top10 = fpl.predict(fpl.current_model, gw)
    return top10.to_dict(orient="records")