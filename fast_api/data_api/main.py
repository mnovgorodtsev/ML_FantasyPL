from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import pandas as pd

app = FastAPI(title="FPL Data API")
df = pd.read_csv("fast_api/data_api/data.csv")

@app.get("/data")
def get_data(
    gw: Optional[int] = Query(None),
    gw_max: Optional[int] = Query(None)
):
    result = df.copy()
    if gw is not None:
        result = result[result["GW"] == gw]
        if result.empty:
            raise HTTPException(status_code=404, detail=f"No data for GW {gw}")
    if gw_max is not None:
        result = result[result["GW"] <= gw_max]
        if result.empty:
            raise HTTPException(status_code=404, detail=f"No data for GW <= {gw_max}")
    return result.to_dict(orient="records")