from datetime import date
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from car_price_pipeline import get_reference_options, predict_price, predict_price_range, train_and_save_model
from market_reference import (
    DEMAND_ADJUSTMENTS,
    MAINTENANCE_ADJUSTMENTS,
    MAHARASHTRA_EX_SHOWROOM,
    CATALOG_LAST_UPDATED,
    adjust_prediction,
    catalog_entry,
    current_market_snapshot,
    market_index,
    refresh_market_snapshot,
)


def render_footer():
    st.markdown(
        """
        <div style="text-align:center; margin-top:3rem; padding:1rem 0;
                    color:#94a3b8; border-top:1px solid rgba(148,163,184,0.2);">
            Made with 🧡 in महाराष्ट्र
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_pdf_report(rows, market_label):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except ImportError as exc:
        raise RuntimeError("PDF export requires the reportlab package.") from exc

    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=36, leftMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Vehicle Value Studio - Prediction Report", styles["Title"]),
        Paragraph(f"Generated: {date.today().isoformat()} | {market_label}", styles["Normal"]),
        Spacer(1, 16),
    ]
    table_rows = [["Vehicle", "Estimated used price", "Price range", "Maharashtra new price"]]
    for row in rows:
        table_rows.append([
            row["name"],
            f"Rs. {row['prediction']:,.0f}",
            f"Rs. {row['lower']:,.0f} - Rs. {row['upper']:,.0f}",
            f"Rs. {row['new_price']:,.0f}" if row["new_price"] else "Not selected",
        ])
    table = Table(table_rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1228ea")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([table, Spacer(1, 16), Paragraph(
        f"Maharashtra reference catalog version: {CATALOG_LAST_UPDATED}. "
        "Prices are indicative starting ex-showroom references; verify current OEM/dealer quotes.",
        styles["Normal"],
    )])
    document.build(story)
    return output.getvalue()


def play_prediction_sound():
    if not st.session_state.get("sound_enabled", False):
        return

    try:
        from playsound3 import playsound
    except ImportError:
        return

    audio_path = Path(__file__).resolve().with_name("car-acceleration-drive.mp3")
    if audio_path.exists():
        playsound(str(audio_path))

st.set_page_config(page_title="Vehicle Value Studio", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stMetric {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            padding: 0.75rem 1rem;
        }
        h1, h2, h3 {
            color: #f8fafc;
        }
        .premium-panel {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(96, 165, 250, 0.25);
            border-radius: 18px;
            padding: 1.3rem;
            backdrop-filter: blur(8px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return train_and_save_model()


@st.cache_data
def load_options():
    return get_reference_options()


options = load_options()
current_year = date.today().year
market = current_market_snapshot()

if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = False

st.title("🚘 Vehicle Value Studio")
st.caption(" Used Car Price estimator | Audio remains optional and off by default")

with st.sidebar:
    st.markdown("### Model Overview")
    st.info("Random Forest regressor trained on a cleaned used-car dataset with engineered market features.")
    st.markdown("**Status:** Ready")
    st.markdown("**Deployment pattern:** persisted model artifact, reusable preprocessing logic")
    st.markdown(f"**Prices refer to:** {market['as_of'].strftime('%B %Y')}")
    if "refreshed_at" in market:
        st.caption(f"Last refreshed: {market['refreshed_at']}")
    else:
        st.warning("Current month has no saved market snapshot yet.")
    market_change_pct = st.slider(
        "Current market movement (%)",
        min_value=-20,
        max_value=25,
        value=round((market["factor"] - 1) * 100),
        help="Enter the verified monthly Maharashtra market movement before saving a refresh.",
    )
    refresh_source = st.text_input(
        "Refresh source",
        value="Verified Maharashtra monthly market update",
    )
    if st.button("Refresh current month prices", use_container_width=True):
        refresh_market_snapshot(market_change_pct, refresh_source)
        st.success(f"Saved market prices for {date.today():%B %Y}.")
        st.rerun()
    market["factor"] = 1 + market_change_pct / 100
    market["label"] = f"Market index {market['as_of'].strftime('%b %Y')} ({market_change_pct:+d}%)"
    st.toggle("Sound effects", key="sound_enabled")
    if st.session_state.sound_enabled:
        st.caption("Audio is enabled.")

left, right = st.columns([1.55, 1])

with left:
    st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
    st.subheader("Car specifications")

    brand_col, year_col = st.columns(2)
    with brand_col:
        brand = st.selectbox("Brand", options["brand"])
    with year_col:
        registration_year = st.slider("Registration year", min_value=options["year_min"], max_value=current_year, value=2018)

    km_col, fuel_col = st.columns(2)
    with km_col:
        km_driven = st.number_input("Kilometers driven", min_value=0, max_value=500000, value=52000, step=1000)
    with fuel_col:
        fuel = st.selectbox("Fuel type", options["fuel"])

    seller_col, transmission_col = st.columns(2)
    with seller_col:
        seller_type = st.selectbox("Seller type", options["seller_type"])
    with transmission_col:
        transmission = st.selectbox("Transmission", options["transmission"])

    model_catalog = [
        model_name
        for model_name, details in MAHARASHTRA_EX_SHOWROOM.items()
        if details["brand"] == brand
    ]
    model_name = st.selectbox(
        "Car model for new-car comparison",
        ["Not selected"] + model_catalog,
        help="Used only for the Maharashtra ex-showroom comparison; the ML model still uses the selected vehicle features.",
    )
    compare_mode = st.checkbox("Enable multi-car comparison")
    comparison_models = st.multiselect(
        "Models to compare (up to 4)",
        list(MAHARASHTRA_EX_SHOWROOM),
        default=[],
        max_selections=4,
        disabled=not compare_mode,
    )

    owner_col, seats_col = st.columns(2)
    with owner_col:
        owner = st.selectbox("Owner history", options["owner"])
    with seats_col:
        seats = st.slider("Seats", min_value=2, max_value=7, value=5)

    engine_col, power_col = st.columns(2)
    with engine_col:
        engine = st.number_input("Engine capacity (CC)", min_value=500, max_value=5000, value=1200, step=50)
    with power_col:
        max_power = st.number_input("Max power (bhp)", min_value=20.0, max_value=400.0, value=75.0, step=2.5)

    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.0, step=0.5)
    maintenance = st.select_slider(
        "Maintenance condition",
        options=list(MAINTENANCE_ADJUSTMENTS),
        value="Average",
        help="Adjusts the estimate for service history and visible condition.",
    )
    demand = st.select_slider(
        "Seller demand",
        options=list(DEMAND_ADJUSTMENTS),
        value="Normal",
        help="Adjusts the estimate for current buyer demand in the selected market.",
    )
    car_age = max(current_year - registration_year, 0)

    predict_button = st.button("Estimate price", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
    st.subheader("Market snapshot")
    st.metric("Dataset size", "8,128 listings")
    st.metric("Model type", "RandomForestRegressor")
    st.metric("Price signal", "Market-based valuation")
    st.metric("Market index", f"{market['factor']:.2f}x")
    st.caption(f"{market['label']}. {market['method']}")
    st.caption(f"Maharashtra reference catalog: {CATALOG_LAST_UPDATED}")
    st.caption("Maintenance and demand are transparent user-provided adjustments.")
    st.markdown("</div>", unsafe_allow_html=True)


if predict_button:
    user_input = {
        "brand": brand,
        "car_age": car_age,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }

    try:
        get_model()
        model_prediction = predict_price(user_input)
        base_lower, base_upper = predict_price_range(user_input)
        adjusted = adjust_prediction(model_prediction, maintenance, demand, market)
        adjustment_factor = adjusted["final_price"] / model_prediction if model_prediction else 1.0
        prediction = adjusted["final_price"]
        lower_bound = max(base_lower * adjustment_factor, 0.0)
        upper_bound = max(base_upper * adjustment_factor, lower_bound)

        st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
        st.subheader("Estimated selling price")
        st.metric("Indicative value", f"₹ {prediction:,.0f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Lower band", f"₹ {lower_bound:,.0f}")
        c2.metric("Upper band", f"₹ {upper_bound:,.0f}")
        c3.metric("Age", f"{car_age} yrs")

        st.success(
            f"Updated for {market['label'].lower()}, {maintenance.lower()} maintenance, "
            f"and {demand.lower()} seller demand."
        )
        st.caption(
            f"Model baseline: ₹ {model_prediction:,.0f} | "
            f"Market adjustment: {adjusted['market_change']:+,.0f}"
        )

        if model_name != "Not selected":
            new_car = catalog_entry(model_name)
            if new_car:
                percentage_of_new = prediction / new_car["price"] * 100
                st.markdown("#### Maharashtra new-car comparison")
                compare_col1, compare_col2 = st.columns(2)
                compare_col1.metric("Indicative ex-showroom", f"₹ {new_car['price']:,.0f}")
                compare_col2.metric("Used price vs new", f"{percentage_of_new:.0f}%")
                st.caption(
                    f"{model_name} reference price is an indicative Maharashtra starting price "
                    f"({new_car['source']}); verify the current dealer/OEM quote before purchase."
                )
        report_rows = [{
            "name": model_name if model_name != "Not selected" else brand,
            "prediction": prediction,
            "lower": lower_bound,
            "upper": upper_bound,
            "new_price": catalog_entry(model_name)["price"] if model_name != "Not selected" else None,
        }]
        if compare_mode and comparison_models:
            comparison_rows = []
            for comparison_model in comparison_models:
                comparison_brand = MAHARASHTRA_EX_SHOWROOM[comparison_model]["brand"]
                comparison_input = {**user_input, "brand": comparison_brand}
                comparison_base = predict_price(comparison_input)
                comparison_adjusted = adjust_prediction(comparison_base, maintenance, demand, market)
                comparison_factor = comparison_adjusted["final_price"] / comparison_base if comparison_base else 1.0
                comparison_rows.append({
                    "Model": comparison_model,
                    "Estimated price": comparison_adjusted["final_price"],
                    "New ex-showroom": MAHARASHTRA_EX_SHOWROOM[comparison_model]["price"],
                    "Used vs new": comparison_adjusted["final_price"] / MAHARASHTRA_EX_SHOWROOM[comparison_model]["price"] * 100,
                })
                report_rows.append({
                    "name": comparison_model,
                    "prediction": comparison_adjusted["final_price"],
                    "lower": max(base_lower * comparison_factor, 0.0),
                    "upper": max(base_upper * comparison_factor, 0.0),
                    "new_price": MAHARASHTRA_EX_SHOWROOM[comparison_model]["price"],
                })
            st.markdown("#### Multi-car comparison")
            st.dataframe(comparison_rows, hide_index=True, use_container_width=True)
        st.download_button(
            "Download prediction report (PDF)",
            data=build_pdf_report(report_rows, market["label"]),
            file_name=f"vehicle-value-report-{date.today().isoformat()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        play_prediction_sound()
        st.markdown("</div>", unsafe_allow_html=True)

        car_html = """
        <div id="car-burst"></div>
        <script>
        const container = document.getElementById('car-burst');
        const cars = ['🚗', '🚙', '🏎️'];
        for (let i = 0; i < 24; i++) {
            const el = document.createElement('span');
            el.textContent = cars[Math.floor(Math.random() * cars.length)];
            el.style.position = 'fixed';
            el.style.left = (Math.random() * 90 + 5) + 'vw';
            el.style.top = '-12vh';
            el.style.fontSize = (22 + Math.random() * 20) + 'px';
            el.style.pointerEvents = 'none';
            el.style.zIndex = '9999';
            el.style.transition = 'transform 2.5s linear, opacity 2.8s ease-out';
            container.appendChild(el);
            requestAnimationFrame(() => {
                el.style.transform = `translateY(${110 + Math.random() * 20}vh) rotate(${Math.random() * 360}deg)`;
                el.style.opacity = '0';
            });
            setTimeout(() => el.remove(), 3200);
        }
        </script>
        """
        components.html(car_html, height=0)

    except Exception as exc:  # pragma: no cover
        st.error(f"Prediction failed: {exc}")
        st.caption("Please verify your inputs are valid and try again.")

render_footer()
