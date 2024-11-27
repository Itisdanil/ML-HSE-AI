from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

app = FastAPI()


with open("model_artifacts/model.pkl",'rb') as f:
    model = pickle.load(f)

with open("model_artifacts/ohe.pkl",'rb') as f:
    ohe = pickle.load(f)


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    torque: float
    max_torque_rpm: float
    seats: float


class Items(BaseModel):
    objects: List[Item]


def transform_features(item_df: pd.DataFrame):
    categorical_columns = ["name", "fuel", "seller_type", "transmission", "owner", "seats"]
    item_encoded = ohe.transform(item_df[categorical_columns])
    item_df = pd.concat([item_df.drop(categorical_columns, axis=1), pd.DataFrame(item_encoded, columns=ohe.get_feature_names_out(categorical_columns))], axis=1)
    return item_df


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_df = pd.DataFrame([item.dict()])
    item_df = transform_features(item_df)
    prediction = model.predict(item_df)
    return float(prediction[0])


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    item_list = items.objects
    item_df = pd.DataFrame([item.dict() for item in item_list])
    item_df = transform_features(item_df)
    predictions = model.predict(item_df)
    return predictions.tolist()


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = transform_features(df)
    predictions = model.predict(df)
    df["predicted_price"] = predictions
    result_file = "data_artifacts/cars_predicted_prices.csv"
    df.to_csv(result_file, index=False)
    return result_file
