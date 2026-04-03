import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


"""
start commands:

mlflow server --host 127.0.0.1 --port 5000

cd fantasyML
uvicorn fast_api.data_api.main:app --reload --port 8000
uvicorn fast_api.model_api.main:app --port 8001

python website_flask/app.py 
"""