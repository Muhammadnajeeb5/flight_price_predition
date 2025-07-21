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
            "Airline", "Destination", "Total_Stops", "Journey_day", "Journey_month",
            "Arrival_hour", "Arrival_minute", "Dep_hour", "Dep_minute",
            "Duration_hours", "Duration_minutes",
            "Source_Banglore", "Source_Kolkata", "Source_Delhi", "Source_Chennai", "Source_Mumbai"
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
        df = pd.read_excel(r"C:\Users\HP\Music\Data_Train.xlsx")
        df.dropna(inplace=True)
        df['Price'] = df['Price'].apply(lambda x: min(x, 35000))
        stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
        df['Total_Stops'] = df['Total_Stops'].map(stops_mapping)
        
        # Plot 1: Airline vs. Price (Box Plot)
        fig1 = px.box(df, x="Airline", y="Price", title="Airline vs. Price",
                      color="Airline", template="plotly_dark")
        plot1_div = plot(fig1, output_type="div", include_plotlyjs=False)
        
        # Plot 2: Total Stops vs. Price (Violin Plot)
        fig2 = px.violin(df, x="Total_Stops", y="Price", box=True, points="all",
                         title="Total Stops vs. Price", template="plotly_dark")
        plot2_div = plot(fig2, output_type="div", include_plotlyjs=False)
        
        # Plot 3: Correlation Heatmap
        df['Journey_day'] = pd.to_datetime(df['Date_of_Journey']).dt.day
        df['Journey_month'] = pd.to_datetime(df['Date_of_Journey']).dt.month
        df['Duration'] = df['Duration'].apply(lambda x: x if ('h' in x and 'm' in x) else '0h 0m')
        df['Duration_hours'] = df['Duration'].apply(lambda x: int(x.split('h')[0]))
        df['Duration_minutes'] = df['Duration'].apply(lambda x: int(x.split(' ')[1].split('m')[0]))
        corr_cols = ['Price', 'Total_Stops', 'Duration_hours', 'Duration_minutes', 'Journey_day', 'Journey_month']
        fig3 = px.imshow(df[corr_cols].corr(), text_auto=True, title="Feature Correlation Heatmap", template="plotly_dark")
        plot3_div = plot(fig3, output_type="div", include_plotlyjs=False)
        
        logger.info("Dashboard plots generated successfully")
        return render_template("dashboard.html", plot1_div=plot1_div, plot2_div=plot2_div, plot3_div=plot3_div)
    
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
                logger.warning("No file uploaded in /explore")
                return redirect(request.url)

            df = pd.read_excel(file)
            logger.info("Raw data loaded successfully")

            # --- Data Preprocessing (Crucial!) ---
            try:
                # Date and Time Features
                df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y', errors='coerce')
                df['Journey_day'] = df['Date_of_Journey'].dt.day
                df['Journey_month'] = df['Date_of_Journey'].dt.month

                df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce') # Assuming HH:MM
                df['Dep_Time_hour'] = df['Dep_Time'].dt.hour
                df['Dep_Time_minute'] = df['Dep_Time'].dt.minute

                df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M', errors='coerce') # Assuming HH:MM
                df['Arrival_Time_hour'] = df['Arrival_Time'].dt.hour
                df['Arrival_Time_minute'] = df['Arrival_Time'].dt.minute

                # Duration
                def preprocess_duration(x):
                    if 'h' not in x:
                        x = '0h ' + x
                    elif 'm' not in x:
                        x = x + ' 0m'
                    return x

                df['Duration'] = df['Duration'].apply(preprocess_duration)
                df[['Duration_hours', 'Duration_minutes']] = df['Duration'].str.split(' ', expand=True)
                df['Duration_hours'] = df['Duration_hours'].str.extract('(\d+)').astype(int)
                df['Duration_minutes'] = df['Duration_minutes'].str.extract('(\d+)').astype(int)
                df.drop(columns=['Duration'], inplace=True)

                # Encode categorical values
                df['Destination'].replace('New Delhi', 'Delhi', inplace=True)
                dest = df.groupby('Destination')['Price'].mean().sort_values().index
                df['Destination'] = df['Destination'].map({k: i for i, k in enumerate(dest)})

                airlines = df.groupby('Airline')['Price'].mean().sort_values().index
                df['Airline'] = df['Airline'].map({k: i for i, k in enumerate(airlines)})

                stops = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
                df['Total_Stops'] = df['Total_Stops'].map(stops)

                for cat in df['Source'].unique():
                    df['Source_' + cat] = df['Source'].apply(lambda x: 1 if x == cat else 0)

                # Drop unused columns
                df.drop(columns=[
                    'Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Route',
                    'Additional_Info', 'Source', 'Journey_year',
                    'Duration_total_mins'
                ], errors='ignore', inplace=True)

                # Outlier cap (Important to apply this!)
                df['Price'] = np.where(df['Price'] > 35000, df['Price'].median(), df['Price'])

                df.dropna(inplace=True)  # Remove any remaining NaNs after processing
                logger.info("Data preprocessing completed")

            except Exception as preprocess_error:
                logger.error(f"Error during preprocessing: {preprocess_error}", exc_info=True)
                flash(f"Error during data preprocessing: {preprocess_error}", "error")
                return redirect(request.url)

            # --- Plotting ---
            try:
                # Plot A: Price Distribution
                figA = px.histogram(df, x="Price", title="Price Distribution", template="plotly_dark")
                plotA_div = plot(figA, output_type="div", include_plotlyjs=False)

                # Plot B: Duration (hours) vs Price if Duration column exists
                if "Duration_hours" in df.columns and "Price" in df.columns:
                    figB = px.scatter(df, x="Duration_hours", y="Price", title="Duration (hours) vs Price", template="plotly_dark")
                    plotB_div = plot(figB, output_type="div", include_plotlyjs=False)
                else:
                    plotB_div = "<p>Duration or Price column not found.</p>"

                # Plot C: Correlation Heatmap
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 1:
                    figC = px.imshow(df[numeric_cols].corr(), text_auto=True, title="Correlation Heatmap", template="plotly_dark")
                    plotC_div = plot(figC, output_type="div", include_plotlyjs=False)
                else:
                    plotC_div = "<p>Not enough numeric columns for correlation plot.</p>"

                logger.info("User-uploaded data processed and plots generated successfully")
                return render_template("explore.html", plotA_div=plotA_div, plotB_div=plotB_div, plotC_div=plotC_div)

            except Exception as plot_error:
                logger.error(f"Error during plotting: {plot_error}", exc_info=True)
                flash(f"Error during plotting: {plot_error}", "error")
                return redirect(request.url)

        except Exception as overall_error:
            logger.error(f"Error processing uploaded file: {overall_error}", exc_info=True)
            flash(f"Error processing file: {overall_error}", "error")
            return redirect(request.url)

    else:
        logger.info("Explore page requested (GET)")
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

        # For proper SHAP explanations, we need a sample of your model’s input features.
        # Ideally, you would use the same preprocessed data that was used during training.
        # Here we simulate that by generating a DataFrame with the same 16 columns (order must match training).
        # Update this section with your actual preprocessed features if available.
        columns = [
            "Airline", "Destination", "Total_Stops", "Journey_day", "Journey_month",
            "Arrival_hour", "Arrival_minute", "Dep_hour", "Dep_minute",
            "Duration_hours", "Duration_minutes",
            "Source_Banglore", "Source_Kolkata", "Source_Delhi", "Source_Chennai", "Source_Mumbai"
        ]
        # For demonstration purposes, generate 50 rows of dummy data.
        # In practice, replace this with your real preprocessed dataset, for example by loading it from a CSV.
        import numpy as np
        X_sample = pd.DataFrame(np.random.rand(50, len(columns)), columns=columns)
        
        # Create a SHAP TreeExplainer (since your model is a Random Forest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Create a SHAP summary plot (a bar plot for clarity)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        
        # Save the plot to a BytesIO object and encode as Base64 to pass to the template
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        
        return render_template("shap.html", plot_url=plot_url)
    
    except Exception as e:
        # Log the error and return a simple error message
        app.logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
        return f"Error generating SHAP explanation: {e}"


if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(debug=True)
