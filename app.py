# app.py

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
import logging


# Fix for matplotlib backend to avoid threading GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import Plotly functions for interactive charts
import plotly.express as px
from plotly.offline import plot

# ----------------------------
# 1. Logging Configuration
# ----------------------------
# Set up basic logging to file (app.log) with INFO level.
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # for flashing messages

# Load the best model
try:
    model = pickle.load(open('final_rf.pkl', 'rb'))
    logger.info("Model loaded successfully from final_rf.pkl")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e  # Stop the app if the model fails to load

# Dropdown options for prediction page (must match training categories)
airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara']
destinations = ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata', 'Banglore']
sources = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']

@app.route('/')
def home():
    logger.info("Home page requested")
    return render_template('home.html', airlines=airlines, destinations=destinations, sources=sources)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        airline = request.form['airline']
        destination = request.form['destination']
        source = request.form['source']
        total_stops = int(request.form['stops'])
        journey_day = int(request.form['journey_day'])
        journey_month = int(request.form['journey_month'])
        arrival_time_hour = int(request.form['arrival_hour'])
        arrival_time_minute = int(request.form['arrival_minute'])
        dep_time_hour = int(request.form['dep_hour'])
        dep_time_minute = int(request.form['dep_minute'])
        duration_hours = int(request.form['duration_hour'])
        duration_minutes = int(request.form['duration_minute'])

        airline_map = {name: idx for idx, name in enumerate(airlines)}
        destination_map = {name: idx for idx, name in enumerate(destinations)}
        airline_encoded = airline_map.get(airline, 0)
        destination_encoded = destination_map.get(destination, 0)

        source_encoded = [0] * len(sources)
        if source in sources:
            source_encoded[sources.index(source)] = 1

        feature_values = [
            airline_encoded, destination_encoded, total_stops,
            journey_day, journey_month, dep_time_hour, dep_time_minute,
            arrival_time_hour, arrival_time_minute, duration_hours, duration_minutes
        ] + source_encoded

        columns = [
            "Airline", "Destination", "Total_Stops", "Journey_day", "Journey_month",
            "Dep_Time_hour", "Dep_Time_minute", "Arrival_Time_hour", "Arrival_Time_minute",
            "Duration_hours", "Duration_minutes",
            "Source_Banglore", "Source_Kolkata", "Source_Delhi", "Source_Chennai", "Source_Mumbai"
        ]

        final_df = pd.DataFrame([feature_values], columns=columns)

        logger.info(f"Feature vector: {feature_values}")
        prediction = model.predict(final_df)[0]
        logger.info(f"Prediction successful: ₹{round(prediction, 2)}")

        return render_template('home.html',
                               prediction_text=f'Estimated Flight Price: ₹{round(prediction, 2)}',
                               airlines=airlines, destinations=destinations, sources=sources)

    except Exception as e:
        logger.error(f"Error in /predict route: {e}", exc_info=True)
        return render_template('home.html',
                               prediction_text=f'Error: {e}',
                               airlines=airlines, destinations=destinations, sources=sources)

# ----------------------------
# Insights: Interactive Feature Importance with Plotly
# ----------------------------
@app.route("/insights")
def insights():
    try:
        logger.info("Insights page requested")
        # Define the feature names (must match the training order)
        feature_names = [
            "Airline", "Destination", "Total_Stops", 
            "Journey_day", "Journey_month",
            "Dep_Time_hour", "Dep_Time_minute",
            "Arrival_Time_hour", "Arrival_Time_minute",
            "Duration_hours", "Duration_minutes",
            "Source_Banglore", "Source_Kolkata", 
            "Source_Delhi", "Source_Chennai", "Source_Mumbai"
        ]
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance", template="plotly_dark")
        plot_div = plot(fig, output_type="div", include_plotlyjs=False)
        logger.info("Insights generated successfully")
        return render_template("insights.html", plot_div=plot_div)
    
    except Exception as e:
        logger.error(f"Error generating insights: {e}", exc_info=True)
        return f"Error generating insights: {e}"

# ----------------------------
# Dashboard: Interactive Visualizations from Training Data
# ----------------------------
@app.route("/dashboard")
def dashboard():
    try:
        logger.info("Dashboard page requested")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(BASE_DIR, "Data_Train.xlsx")

        if not os.path.exists(data_path):
            return "Error: Data_Train.xlsx not found in project folder"

        df = pd.read_excel(data_path)
        df.dropna(inplace=True)
        df['Price'] = df['Price'].apply(lambda x: min(x, 35000))

        fig1 = px.box(
            df, x="Airline", y="Price",
            title="Airline vs. Price",
            color="Airline", template="plotly_dark"
        )
        plot1_div = plot(fig1, output_type="div", include_plotlyjs=False)

        logger.info("Dashboard plot generated successfully")
        return render_template(
            "dashboard.html",
            plot1_div=plot1_div,
            plot2_div="",
            plot3_div=""
        )

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}", exc_info=True)
        return f"Error generating dashboard: {e}"
# ----------------------------
# Explore: User-Uploaded Dataset for Exploration
# ----------------------------
@app.route("/explore", methods=["GET", "POST"])
def explore():
    if request.method == "POST":
        try:
            logger.info("File upload received in /explore")
            file = request.files["datafile"]
            if not file:
                flash("No file selected", "error")
                return redirect(request.url)

            # Load raw data — no encoding needed for visualisation
            df = pd.read_excel(file)
            df.dropna(inplace=True)

            # Cap extreme prices for cleaner charts
            df['Price'] = df['Price'].apply(lambda x: min(x, 35000))

            logger.info(f"Data loaded: {df.shape[0]} rows")
            logger.info(f"Price sample: {df['Price'].head().tolist()}")


            # ── Plot B: Airline vs Price ──────────────────────────
            figB = px.box(
                df, x="Airline", y="Price",
                title="Airline vs Price",
                color="Airline",
                template="plotly_dark"
            )
            plotB_div = plot(figB, output_type="div", include_plotlyjs=False)

            # ── Plot C: Total Stops vs Price ──────────────────────
            figC = px.box(
                df, x="Total_Stops", y="Price",
                title="Total Stops vs Price",
                color="Total_Stops",
                template="plotly_dark"
            )
            plotC_div = plot(figC, output_type="div", include_plotlyjs=False)

            logger.info("Explore plots generated successfully")
            return render_template(
                "explore.html",
                plotB_div=plotB_div,
                plotC_div=plotC_div
            )

        except Exception as e:
            logger.error(f"Error in explore route: {e}", exc_info=True)
            flash(f"Error processing file: {e}", "error")
            return redirect(request.url)

    else:
        return render_template("explore.html")
# ----------------------------
# SHAP: Model Explainability Route
# ----------------------------
@app.route("/shap")
def shap_explain():
    try:
        import shap
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        logger.info("SHAP explanation requested")

        # Column names must match exactly what the model was trained on
        columns = [
            "Airline", "Destination", "Total_Stops", "Journey_day", "Journey_month",
            "Dep_Time_hour", "Dep_Time_minute", "Arrival_Time_hour", "Arrival_Time_minute",
            "Duration_hours", "Duration_minutes",
            "Source_Banglore", "Source_Kolkata", "Source_Delhi", "Source_Chennai", "Source_Mumbai"
        ]

        # Load real training data saved from the notebook
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(BASE_DIR, "X_train_sample.csv")

        if os.path.exists(sample_path):
            # Use real training data — this gives meaningful SHAP values
            X_sample = pd.read_csv(sample_path)

            # Keep only the columns the model expects, in the right order
            X_sample = X_sample[[col for col in columns if col in X_sample.columns]]

            # Take a random sample of 50 rows for speed
            X_sample = X_sample.sample(min(50, len(X_sample)), random_state=42)
            logger.info("Loaded real training data for SHAP")

        else:
            # Fallback: use dummy data if CSV not found, but warn about it
            logger.warning("X_train_sample.csv not found — using dummy data for SHAP")
            X_sample = pd.DataFrame(
                np.zeros((50, len(columns))), 
                columns=columns
            )

        # Create SHAP TreeExplainer — designed specifically for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Generate SHAP summary bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            plot_type="bar", 
            show=False,
            plot_size=(10, 6)
        )
        plt.title("SHAP Feature Importance", color="white", fontsize=14)
        plt.tight_layout()

        # Convert plot to base64 image to embed directly in HTML
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", 
                    facecolor="#1c1c2b", dpi=100)
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        logger.info("SHAP explanation generated successfully")
        return render_template("shap.html", plot_url=plot_url)

    except ImportError:
        logger.error("SHAP library not installed")
        return "Error: SHAP library is not installed. Run: pip install shap"

    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
        return f"Error generating SHAP explanation: {e}"
        
if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(debug=True)