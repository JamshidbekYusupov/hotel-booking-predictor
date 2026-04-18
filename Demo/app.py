import os
import sys
import joblib
import pandas as pd
import gradio as gr

# =========================
# Load model
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

model_path = os.path.join(PROJECT_ROOT, "XGBoost.joblib")

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model from {model_path}: {e}")


# =========================
# Prediction function
# =========================
def predict_booking(
    hotel,
    lead_time,
    arrival_date_year,
    arrival_date_month,
    arrival_date_week_number,
    arrival_date_day_of_month,
    stays_in_weekend_nights,
    stays_in_week_nights,
    adults,
    children,
    babies,
    meal,
    country,
    market_segment,
    distribution_channel,
    is_repeated_guest,
    previous_cancellations,
    previous_bookings_not_canceled,
    reserved_room_type,
    assigned_room_type,
    booking_changes,
    deposit_type,
    agent,
    company,
    days_in_waiting_list,
    customer_type,
    adr,
    required_car_parking_spaces,
    total_of_special_requests,
    reservation_status,
    reservation_status_date,
    city,
):
    try:
        input_data = {
            "hotel": hotel,
            "lead_time": int(lead_time),
            "arrival_date_year": int(arrival_date_year),
            "arrival_date_month": arrival_date_month,
            "arrival_date_week_number": int(arrival_date_week_number),
            "arrival_date_day_of_month": int(arrival_date_day_of_month),
            "stays_in_weekend_nights": int(stays_in_weekend_nights),
            "stays_in_week_nights": int(stays_in_week_nights),
            "adults": int(adults),
            "children": float(children),
            "babies": int(babies),
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": int(is_repeated_guest),
            "previous_cancellations": int(previous_cancellations),
            "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
            "reserved_room_type": reserved_room_type,
            "assigned_room_type": assigned_room_type,
            "booking_changes": int(booking_changes),
            "deposit_type": deposit_type,
            "agent": float(agent),
            "company": float(company),
            "days_in_waiting_list": int(days_in_waiting_list),
            "customer_type": customer_type,
            "adr": float(adr),
            "required_car_parking_spaces": int(required_car_parking_spaces),
            "total_of_special_requests": int(total_of_special_requests),
            "reservation_status": reservation_status,
            "reservation_status_date": reservation_status_date,
            "city": city,
        }

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

        pred_str = str(prediction).strip()
        is_canceled = pred_str in ["1", "1.0"]

        label = "Canceled" if is_canceled else "Not Canceled"
        emoji = "❌" if is_canceled else "✅"
        confidence_note = "This booking is likely to be canceled." if is_canceled else "This booking is likely to remain active."

        result_md = f"""
## {emoji} Prediction Result

**Predicted Class:** {label}

**Interpretation:** {confidence_note}
"""

        return label, result_md

    except Exception as e:
        return "Error", f"## ⚠️ Prediction Error\n\n{str(e)}"


def clear_outputs():
    return "", ""


css = """
.gradio-container {
    max-width: 1200px !important;
}
.main-title {
    text-align: center;
    margin-bottom: 6px;
}
.sub-text {
    text-align: center;
    color: #666;
    margin-bottom: 20px;
}
.section-box {
    border-radius: 14px;
    padding: 14px;
    border: 1px solid #e5e7eb;
    background: #fafafa;
}
.result-box {
    border-radius: 14px;
    padding: 18px;
    border: 1px solid #dbeafe;
    background: #f8fbff;
}
.footer-note {
    font-size: 13px;
    color: #666;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Hotel Booking Cancellation Predictor") as demo:
    gr.HTML("""
        <h1 class="main-title">🏨 Hotel Booking Cancellation Predictor</h1>
        <p class="sub-text">
            Fill in the booking details below and predict whether the reservation is likely to be canceled.
        </p>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="section-box"):
                gr.Markdown("### Guest & Stay Information")

                with gr.Row():
                    hotel = gr.Dropdown(
                        label="Hotel",
                        choices=["Resort Hotel", "City Hotel"],
                        value="City Hotel"
                    )
                    customer_type = gr.Dropdown(
                        label="Customer Type",
                        choices=["Transient", "Contract", "Transient-Party", "Group"],
                        value="Transient"
                    )

                with gr.Row():
                    adults = gr.Number(label="Adults", value=2, precision=0)
                    children = gr.Number(label="Children", value=0)
                    babies = gr.Number(label="Babies", value=0, precision=0)

                with gr.Row():
                    stays_in_weekend_nights = gr.Number(label="Weekend Nights", value=1, precision=0)
                    stays_in_week_nights = gr.Number(label="Week Nights", value=2, precision=0)
                    required_car_parking_spaces = gr.Number(label="Parking Spaces", value=0, precision=0)

            with gr.Group(elem_classes="section-box"):
                gr.Markdown("### Arrival Information")

                with gr.Row():
                    arrival_date_year = gr.Number(label="Arrival Year", value=2017, precision=0)
                    arrival_date_month = gr.Dropdown(
                        label="Arrival Month",
                        choices=[
                            "January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"
                        ],
                        value="July"
                    )

                with gr.Row():
                    arrival_date_week_number = gr.Number(label="Arrival Week Number", value=27, precision=0)
                    arrival_date_day_of_month = gr.Number(label="Arrival Day", value=15, precision=0)
                    lead_time = gr.Number(label="Lead Time", value=50, precision=0)

            with gr.Group(elem_classes="section-box"):
                gr.Markdown("### Booking & Business Details")

                with gr.Row():
                    meal = gr.Textbox(label="Meal", value="BB")
                    country = gr.Textbox(label="Country", value="PRT")
                    city = gr.Textbox(label="City", value="Lisbon")

                with gr.Row():
                    market_segment = gr.Textbox(label="Market Segment", value="Online TA")
                    distribution_channel = gr.Textbox(label="Distribution Channel", value="TA/TO")
                    deposit_type = gr.Dropdown(
                        label="Deposit Type",
                        choices=["No Deposit", "Refundable", "Non Refund"],
                        value="No Deposit"
                    )

                with gr.Row():
                    reserved_room_type = gr.Textbox(label="Reserved Room Type", value="A")
                    assigned_room_type = gr.Textbox(label="Assigned Room Type", value="A")
                    adr = gr.Number(label="ADR", value=100.0)

            with gr.Group(elem_classes="section-box"):
                gr.Markdown("### Customer History & Reservation Status")

                with gr.Row():
                    is_repeated_guest = gr.Number(label="Repeated Guest (0 or 1)", value=0, precision=0)
                    previous_cancellations = gr.Number(label="Previous Cancellations", value=0, precision=0)
                    previous_bookings_not_canceled = gr.Number(
                        label="Previous Non-Canceled Bookings", value=0, precision=0
                    )

                with gr.Row():
                    booking_changes = gr.Number(label="Booking Changes", value=0, precision=0)
                    agent = gr.Number(label="Agent", value=9)
                    company = gr.Number(label="Company", value=0)

                with gr.Row():
                    days_in_waiting_list = gr.Number(label="Days in Waiting List", value=0, precision=0)
                    total_of_special_requests = gr.Number(label="Special Requests", value=1, precision=0)
                    reservation_status = gr.Textbox(label="Reservation Status", value="Check-Out")

                reservation_status_date = gr.Textbox(
                    label="Reservation Status Date (YYYY-MM-DD)",
                    value="2017-07-15"
                )

            with gr.Row():
                predict_btn = gr.Button("Predict Booking Status", variant="primary", size="lg")
                clear_btn = gr.Button("Clear Output", variant="secondary")

        with gr.Column(scale=2):
            with gr.Group(elem_classes="result-box"):
                gr.Markdown("### Prediction Output")
                predicted_class = gr.Textbox(label="Predicted Class", interactive=False)
                result_markdown = gr.Markdown(
                    value="## Waiting for input\n\nEnter booking details and click **Predict Booking Status**."
                )

                gr.Markdown(
                    """
                    <div class="footer-note">
                    Note: If your saved model contains custom pipeline objects from your project,
                    those modules must be importable when loading the model.
                    </div>
                    """
                )

    predict_btn.click(
        fn=predict_booking,
        inputs=[
            hotel,
            lead_time,
            arrival_date_year,
            arrival_date_month,
            arrival_date_week_number,
            arrival_date_day_of_month,
            stays_in_weekend_nights,
            stays_in_week_nights,
            adults,
            children,
            babies,
            meal,
            country,
            market_segment,
            distribution_channel,
            is_repeated_guest,
            previous_cancellations,
            previous_bookings_not_canceled,
            reserved_room_type,
            assigned_room_type,
            booking_changes,
            deposit_type,
            agent,
            company,
            days_in_waiting_list,
            customer_type,
            adr,
            required_car_parking_spaces,
            total_of_special_requests,
            reservation_status,
            reservation_status_date,
            city,
        ],
        outputs=[predicted_class, result_markdown]
    )

    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[predicted_class, result_markdown]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)