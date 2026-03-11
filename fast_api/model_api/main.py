from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from model.model_class import FPLModel

fpl = FPLModel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    fpl.train(13)
    fpl.train_production(current_gw=13)
    yield

app = FastAPI(title="FPL Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(gw: int = Query(...)):
    top10 = fpl.predict(fpl.current_model, gw)
    return top10.to_dict(orient="records")