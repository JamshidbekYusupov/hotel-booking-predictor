from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

model_path = r'C:\hotel_booking\XGBoost.joblib'

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


app = FastAPI(
    title="Let's find together whether this room will be canceled or not :)",
    version='1.0.0',
    description='Hotel Canceletion Classifier Model API'
) 

class Item(BaseModel):
    hotel:object
    lead_time:int
    arrival_date_year:int
    arrival_date_month:object
    arrival_date_week_number:int
    arrival_date_day_of_month:int
    stays_in_weekend_nights:int
    stays_in_week_nights:int
    adults:int
    children:float
    babies:int
    meal:object
    country:object
    market_segment:object
    distribution_channel:object
    is_repeated_guest:int
    previous_cancellations:int
    previous_bookings_not_canceled:int
    reserved_room_type:object
    assigned_room_type:object
    booking_changes:int
    deposit_type:object
    agent:float
    company:float
    days_in_waiting_list:int
    customer_type:object
    adr:float
    required_car_parking_spaces:int
    total_of_special_requests:int
    reservation_status:object
    reservation_status_date:object
    city:object

@app.get("/")
def root():
    return {"message": "Welcome to the Hotel classifier model"}

@app.get("/health/")
def health():
    return {"Status": "API is running"}

@app.post("/XGBooost cassifier/")
def predict(item:Item):
    try:
        ## Converting input to dictionary
        input = item.model_dump()
        ## Converting dictionary to df
        df = pd.DataFrame([input])

        ## Predction 

        prediction = model.predict(df)[0]
        
        return {
            "Predicted Class": int(prediction) if str(prediction).isdigit else str(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
