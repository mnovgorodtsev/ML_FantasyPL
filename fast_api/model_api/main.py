from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from model.model_class import FPLModel
import mlflow

fpl = FPLModel()
training_status = {"is_training": False, "gw": None, "error": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="FPL Model API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train")
def train(gw: int = Query(...), background_tasks: BackgroundTasks = None):
    if training_status["is_training"]:
        return {"status": "already_training", "gw": training_status["gw"]}

    def _run():
        training_status["is_training"] = True
        training_status["error"] = None
        try:
            fpl.hyperopt(gw)
            fpl.current_model_params = fpl.best_params
            fpl.train_production(current_gw=gw)
            fpl.current_gw = gw
        except Exception as e:
            training_status["error"] = str(e)
        finally:
            training_status["is_training"] = False

    background_tasks.add_task(_run)
    return {"status": "started", "gw": gw}

@app.get("/train/status")
def get_training_status():
    return training_status

@app.get("/predict")
def predict(gw: int = Query(...)):
    if training_status["is_training"]:
        return {"error": "Model is still training"}
    if fpl.current_model is None:
        return {"error": "Model not trained yet. Call POST /train?gw=X first."}
    top = fpl.predict(fpl.current_model, gw)
    return top.to_dict(orient="records")