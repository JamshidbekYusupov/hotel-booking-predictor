from sklearn.ensemble import RandomForestClassifier
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

log_path = r'C:\hotel_booking\Logging\engineering.log'

logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)

class HotelFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, apply_log_transform=True):
        self.apply_log_transform = apply_log_transform
        self.drop_cols = [
            'reservation_status',
            'arrival_date_year',
            'assigned_room_type'
        ]
        self.skewed_cols = [
            'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
            'previous_bookings_not_canceled', 'booking_changes', 'agent',
            'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
            'total_guests', 'price_per_person'
        ]

    def fit(self, X, y=None):
        # Nothing to learn from data (stateless transformer)
        return self

    def transform(self, X):
        # Working on a copy to prevent SettingWithCopy warnings
        X = X.copy()
        
        try:
        
            X['total_guests'] = X['adults'] + X['children'] + X['babies']
            X['price_per_person'] = X['adr'] / (X['total_guests'] + 1)

            # 2. LOG TRANSFORM SECOND
            if self.apply_log_transform:
                for col in self.skewed_cols:
                    if col in X.columns:
                        X[col] = np.log1p(X[col])
            
            # 3. Drop unnecessary columns
            X = X.drop(columns=self.drop_cols, errors='ignore')
            logging.info(f'Feature engineering is done successfully')
            
            return X
            
        except Exception as e:
            logging.error(f"Error in Feature Engineering: {e}")
            raise e

    def get_feature_names_out(self, input_features=None):
        # This is for ColumnTransformer or SHAP
        return [col for col in self.feature_names_out_ if col not in self.drop_cols]