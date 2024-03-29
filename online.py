import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

# from scipy.special import expit

st.set_page_config(page_title="Raw Materials Searching System", page_icon=":building_construction:")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Function to download pickle files from GitHub
@st.cache_resource
def download_pickle_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download pickle file from {url}")

# Load the sand model and data
sand_model_url = 'https://github.com/MH-ML/Row-Material/raw/main/sand_model.pkl'
sand_model = download_pickle_from_github(sand_model_url)

sand_X_supplier_url = 'https://github.com/MH-ML/Row-Material/raw/main/sand_X_supplier.pkl'
sand_X_supplier = download_pickle_from_github(sand_X_supplier_url)

# Load the dredging model and data
dredging_model_url = 'https://github.com/MH-ML/Row-Material/raw/main/dridging_model1.pkl'
dredging_model = download_pickle_from_github(dredging_model_url)

dredging_X_supplier_url = 'https://github.com/MH-ML/Row-Material/raw/main/dridging_supplier1.pkl'
dredging_X_supplier = download_pickle_from_github(dredging_X_supplier_url)

# Load the clay brick model and data
clay_model_url = 'https://github.com/MH-ML/Row-Material/raw/main/clay_model1.pkl'
clay_model = download_pickle_from_github(clay_model_url)

clay_X_supplier_url = 'https://github.com/MH-ML/Row-Material/raw/main/claybrick_supplier.pkl'
clay_X_supplier = download_pickle_from_github(clay_X_supplier_url)

scaler_url = 'https://github.com/MH-ML/Row-Material/raw/main/scaler_clay.pkl'
scaler = download_pickle_from_github(scaler_url)

# Load the PPC cement model and data
ppc_model_url = 'https://github.com/MH-ML/Row-Material/raw/main/ppc_model.pkl'
ppc_model = download_pickle_from_github(ppc_model_url)

ppc_scaler_url = 'https://github.com/MH-ML/Row-Material/raw/main/scaler_ppc.pkl'
ppc_scaler = download_pickle_from_github(ppc_scaler_url)

ppc_poly_url = 'https://github.com/MH-ML/Row-Material/raw/main/poly.pkl'
ppc_poly = download_pickle_from_github(ppc_poly_url)

ppc_X_supplier_url = 'https://github.com/MH-ML/Row-Material/raw/main/ppc_supplier.pkl'
ppc_X_supplier = download_pickle_from_github(ppc_X_supplier_url)

# Load the tmt rod trained model
model_url = 'https://github.com/MH-ML/Row-Material/raw/main/rod_model.pkl'
model = download_pickle_from_github(model_url)

# Load the tmt rod supplier data
X_supplier_url = 'https://github.com/MH-ML/Row-Material/raw/main/rod_supplier.pkl'
X_supplier = download_pickle_from_github(X_supplier_url)


# Sand matching functions
def predict_sand_match(buyer_price, buyer_availability, buyer_clay_content, buyer_clay_lump, buyer_fm, num_predictions=4):
    predictions = []
    suppliers = []

    for _ in range(num_predictions):
        # Get a random supplier
        supplier = sand_X_supplier.sample(1).squeeze().to_dict()

        # Make prediction
        prediction_input = np.array([[buyer_price, buyer_availability, buyer_clay_content, buyer_clay_lump,
                                       supplier['Supplier_Price'], supplier['Supplier_Availability'], supplier['Clay_Content_%'],
                                       supplier['Clay_Lump_%'], buyer_fm, supplier['Supplier_FM']]])
        prediction = sand_model.predict(prediction_input)
        predictions.append(prediction[0])
        suppliers.append(supplier)

    return predictions, suppliers

# Dredging sand matching functions
def predict_dredging_match(buyer_price, buyer_availability, buyer_fm, num_predictions=4):
    predictions = []
    suppliers = []

    for _ in range(num_predictions):
        # Get a random supplier
        supplier = dredging_X_supplier.sample(1).squeeze().to_dict()

        # Make prediction
        prediction_input = np.array([[buyer_price, buyer_availability, buyer_fm,
                                       supplier['Supplier_Price'], supplier['Supplier_Availability'], supplier['FM_y']]])
        prediction = dredging_model.predict(prediction_input) * 100
        predictions.append(prediction[0])
        suppliers.append(supplier)

    return predictions, suppliers

# Clay brick matching functions
def predict_clay_match(buyer_Quality, buyer_price, buyer_availability, num_predictions=4):
    predictions = []
    suppliers = []
    predicted_classes = []
    predicted_probs = []
    for _ in range(num_predictions):
        # Get a random supplier
        supplier = clay_X_supplier.sample(1).squeeze().to_dict()
        # Make prediction
        prediction_input = np.array([[buyer_Quality, buyer_price, buyer_availability, supplier['Supplier_Quality'], supplier['Supplier_Price'], supplier['Supplier_Availability']]])
        prediction_input = scaler.transform(prediction_input)
        prediction_prob = clay_model.predict_proba(prediction_input)[0]
        # Apply sigmoid function to smooth out probabilities
        prediction_prob = sigmoid(prediction_prob) * 100
        prediction = 1 if prediction_prob[1] >= 50 else 0  # Apply threshold for classification
        predictions.append(prediction)
        suppliers.append(supplier)
        predicted_classes.append(prediction)
        predicted_probs.append(prediction_prob)
    return predictions, suppliers, predicted_classes, predicted_probs

# PPC cement matching functions
def predict_ppc_match(buyer_price, buyer_availability, buyer_psi, buyer_pozzolanic_material, buyer_gypsum, buyer_clinker, buyer_slag_fly_ash, buyer_limestone, num_predictions=4):
    predictions = []
    suppliers = []
    for _ in range(num_predictions):
        # Get a random supplier
        supplier = ppc_X_supplier.sample(1).squeeze().to_dict()

        # Make prediction
        prediction_input = np.array([[
            buyer_price, buyer_availability, buyer_psi,
            buyer_pozzolanic_material, buyer_gypsum, buyer_clinker,
            buyer_slag_fly_ash, buyer_limestone,
            supplier['Supplier_Price'], supplier['Supplier_Availability'],
            supplier['PSI_y'], supplier['Pozzolanic_Material%_y'],
            supplier['Gypsum%_y'], supplier['Clinker%_y'],
            supplier['Slag/Fly_Ash%_y'], supplier['Limestone%_y']
        ]])

        # Preprocess the input features
        prediction_input = ppc_scaler.transform(prediction_input)
        prediction_input = ppc_poly.transform(prediction_input)

        prediction = ppc_model.predict(prediction_input)
        predictions.append(prediction[0])
        suppliers.append(supplier)

    return predictions, suppliers

def predict_match(buyer_price, buyer_availability, buyer_steel_rod_grade, buyer_yield_strength, buyer_tensile_strength, buyer_elongation, buyer_sulphur, buyer_carbon, buyer_silicon, num_predictions=4):
    predictions = []
    suppliers = []
    predicted_classes = []
    predicted_probs = []
    for _ in range(num_predictions):
        # Get a random supplier
        supplier = X_supplier.sample(1).squeeze().to_dict()
        # Prepare the input features
        input_features = [buyer_price, buyer_availability, buyer_steel_rod_grade, buyer_yield_strength, buyer_tensile_strength, buyer_elongation, buyer_sulphur, buyer_carbon, buyer_silicon, supplier['Supplier_Price'], supplier['Supplier_Availability'], supplier['Supplier_Steel_Rod_Grade(W)'], supplier['Supplier_Yield_Strength_Mpa'], supplier['Supplier_Tensile_Strength_Mpa'], supplier['Supplier_Elongation'], supplier['Sulphar_y'], supplier['Supplier_Average_Carbon_%'], supplier['Supplier_Average_Silicon_%']]
        prediction_input = np.array([input_features])
        prediction_prob = model.predict_proba(prediction_input)[0]
        prediction_prob = sigmoid(prediction_prob) * 100
        prediction = 1 if prediction_prob[1] >= 50 else 0  # Apply threshold for classification
        predictions.append(prediction)
        suppliers.append(supplier)
        predicted_classes.append(prediction)
        predicted_probs.append(prediction_prob)
    return predictions, suppliers, predicted_classes, predicted_probs

# Streamlit UI
def main():

    st.title("List Of Suppliers")

    app_selection = st.sidebar.selectbox("Select Material", ["Sand", "Dredging Sand", "Clay Brick", "PPC Cement", "TMT Rod"])

    if app_selection == "Sand":
        st.header("Sand Matching Between Buyer And Supplier")

        # Input fields for user
        st.sidebar.header("Enter Buyer Details")
        buyer_details = {
            "Price": st.sidebar.number_input("Price (500 to 10000 standard)"),
            "Availability": st.sidebar.number_input("Availability (200 to 5000 standaed)"),
            "Clay Content (%)": st.sidebar.number_input("Clay Content (%) (1 to 50 standard)"),
            "Clay Lump (%)": st.sidebar.number_input("Clay Lump (%) (0.1 to 0.30 standard)"),
            "FM": st.sidebar.number_input("FM (0 to 30 standard)")
        }

        # Prediction button
        if st.sidebar.button("Search"):
            with st.spinner('Generating Results...'):
                predictions, suppliers = predict_sand_match(
                    buyer_details["Price"],
                    buyer_details["Availability"],
                    buyer_details["Clay Content (%)"],
                    buyer_details["Clay Lump (%)"],
                    buyer_details["FM"]
                )

                st.write("---")

                st.subheader("Dashboard:")

                for i, (prediction, supplier) in enumerate(zip(predictions, suppliers), start=1):
                    st.write(f"Search List {i}:")
                    if prediction >= 50:
                        st.success(f"KPI Score: {prediction:.2f}%")
                        st.subheader("Buyer Details:")
                        st.info("Price: {}\nAvailability: {}\nClay Content (%): {}\nClay Lump (%): {}\nFM: {}".format(
                            buyer_details["Price"],
                            buyer_details["Availability"],
                            buyer_details["Clay Content (%)"],
                            buyer_details["Clay Lump (%)"],
                            buyer_details["FM"]
                        ))

                        st.subheader("Supplier Details:")
                        st.success("Supplier Price: {}\nSupplier Availability: {}\nSupplier Clay Content (%): {}\nSupplier Clay Lump (%): {}\nSupplier FM: {}".format(
                            supplier['Supplier_Price'],
                            supplier['Supplier_Availability'],
                            supplier['Clay_Content_%'],
                            supplier['Clay_Lump_%'],
                            supplier['Supplier_FM']
                        ))

                        st.subheader("Overall KPI Match:")
                        st.success(f"Overall KPI Match between Buyer and Supplier: {prediction:.2f}%")

                    else:
                        st.error(f"No match found. KPI Match Score: {prediction:.2f}%")

                    st.write("---")

    elif app_selection == "Dredging Sand":
        st.header("Dredging Sand Matching Between Buyer And Supplier")

        # Input fields for user
        st.sidebar.header("Enter Buyer Details")
        buyer_details = {
            "Price": st.sidebar.number_input("Price (1000 to 1899 standard)"),
            "Availability": st.sidebar.number_input("Availability ( 800 to 1700 standard)"),
            "FM": st.sidebar.number_input("FM (0.01 to 5.0)")
        }

        # Prediction button
        if st.sidebar.button("Search"):
            with st.spinner('Generating Results...'):
                predictions, suppliers = predict_dredging_match(
                    buyer_details["Price"],
                    buyer_details["Availability"],
                    buyer_details["FM"]
                )

                st.write("---")

                st.subheader("Dashboard:")

                for i, (prediction, supplier) in enumerate(zip(predictions, suppliers), start=1):
                        
                    st.write(f"Search List {i}:")
                    if prediction > 100.00:
                        st.error("No supplier information available for the given inputs.")
                    elif prediction >= 50:
                        st.success(f"KPI Score: {prediction:.2f}%")
                        st.subheader("Buyer Details:")
                        st.info("Price: {}\nAvailability: {}\nFM: {}".format(
                            buyer_details["Price"],
                            buyer_details["Availability"],
                            buyer_details["FM"]
                        ))

                        st.subheader("Supplier Details:")
                        st.success("Supplier Price: {}\nSupplier Availability: {}\nSupplier FM: {}".format(
                            supplier['Supplier_Price'],
                            supplier['Supplier_Availability'],
                            supplier['FM_y']
                        ))

                        st.subheader("Overall KPI Match:")
                        st.success(f"Overall KPI Match between Buyer and Supplier: {prediction:.2f}%")

                    else:
                        st.error(f"No match found. Match Score: {prediction:.2f}%")

                    st.write("---")

    elif app_selection == "Clay Brick":
        st.header("Clay Brick Matching Between Buyer And Supplier")

        # Input fields for user
        st.sidebar.header("Enter Buyer Details")
        buyer_details = {
            "Quality": st.sidebar.number_input("Input Grade (1 for Grade A 2 for Grade B)"),
            "Price": st.sidebar.number_input("Price (500 to)"),
            "Availability": st.sidebar.number_input("Availability (700 to 500 standard)"),
        }

        # Prediction button
        if st.sidebar.button("Search"):
            with st.spinner('Generating Results...'):
                predictions, suppliers, predicted_classes, predicted_probs = predict_clay_match(
                    buyer_details["Quality"], buyer_details["Price"], buyer_details["Availability"])

            st.write("---")
            st.subheader("Dashboard:")
            for i, (prediction, supplier, predicted_class, predicted_prob) in enumerate(zip(predictions, suppliers, predicted_classes, predicted_probs), start=1):
                st.write(f"Search List {i}:")
                if predicted_class == 1:
                    # If predicted class is 1, there's a match
                    st.success(f"KPI Score: {predicted_prob[1]:.0f}%")
                    st.subheader("Buyer Details:")
                    st.info("Quality: {}\nPrice: {}\nAvailability: {}".format(
                        buyer_details["Quality"], buyer_details["Price"], buyer_details["Availability"]))
                    st.subheader("Supplier Details:")
                    st.success("Supplier Quality: {}\nSupplier Price: {}\nSupplier Availability: {}".format(
                        supplier['Supplier_Quality'], supplier['Supplier_Price'], supplier['Supplier_Availability']))
                    st.subheader("Overall KPI Match:")
                    st.success(f"Overall KPI Match between Buyer and Supplier: {predicted_prob[1]:.0f}%")
                else:
                    # If predicted class is 0, there's no match
                    no_match_prob = 1 - predicted_prob[0] * 100
                    st.error(f"No match found. Match Score: {no_match_prob:.0f}%")
                st.write("---")

    elif app_selection == "PPC Cement":
        st.header("PPC Cement Matching Between Buyer And Supplier")

        # Input fields for user
        st.sidebar.header("Enter Buyer Details")
        buyer_details = {
            "Price": st.sidebar.number_input("Price (405 to 494 standard)"),
            "Availability": st.sidebar.number_input("Availability (201 to 400 standard)"),
            "PSI": st.sidebar.number_input("PSI (5001 to 6998 standard)"),
            "Pozzolanic Material %": st.sidebar.number_input("Pozzolanic Material % (0 to 100 %)"),
            "Gypsum %": st.sidebar.number_input("Gypsum % (5 to 20 standard)"),
            "Clinker %": st.sidebar.number_input("Clinker % (70 to 75% Standard)"),
            "Slag/Fly Ash %": st.sidebar.number_input("Slag/Fly Ash %(0 to 6% standard)"),
            "Limestone %": st.sidebar.number_input("Limestone % (20 to 30% standard)")
        }

        # Prediction button
        if st.sidebar.button("Search"):
            with st.spinner('Generating Results...'):
                predictions, suppliers = predict_ppc_match(
                    buyer_details["Price"], buyer_details["Availability"], buyer_details["PSI"],
                    buyer_details["Pozzolanic Material %"], buyer_details["Gypsum %"],
                    buyer_details["Clinker %"], buyer_details["Slag/Fly Ash %"],
                    buyer_details["Limestone %"]
                )

            st.write("---")
            st.subheader("Dashboard:")
            for i, (prediction, supplier) in enumerate(zip(predictions, suppliers), start=1):
                st.write(f"Search List {i}:")
                if prediction >= 70:
                    st.success(f"KPI Score: {prediction:.2f}%")
                    st.subheader("Buyer Details:")
                    st.info(
                        "Price: {}\nAvailability: {}\nPSI: {}\nPozzolanic Material %: {}\nGypsum %: {}\nClinker %: {}\nSlag/Fly Ash %: {}\nLimestone %: {}".format(
                            buyer_details["Price"], buyer_details["Availability"], buyer_details["PSI"],
                            buyer_details["Pozzolanic Material %"], buyer_details["Gypsum %"],
                            buyer_details["Clinker %"], buyer_details["Slag/Fly Ash %"],
                            buyer_details["Limestone %"]
                        )
                    )
                    st.subheader("Supplier Details:")
                    st.success(
                    "Supplier Price: {}\nSupplier Availability: {}\nSupplier PSI: {}\nSupplier Pozzolanic Material %: {}\nSupplier Gypsum %: {}\nSupplier Clinker %: {}\nSupplier Slag/Fly Ash %: {}\nSupplier Limestone %: {}".format(
                        supplier['Supplier_Price'], supplier['Supplier_Availability'], supplier['PSI_y'],
                        supplier['Pozzolanic_Material%_y'], supplier['Gypsum%_y'], supplier['Clinker%_y'],
                        supplier['Slag/Fly_Ash%_y'], supplier['Limestone%_y']
                    )
                )
                st.subheader("Overall KPI Match:")
                st.success(f"Overall KPI Match between Buyer and Supplier: {prediction:.2f}%")
            else:
                st.error(f"No match found. Match Score: {prediction:.2f}%")
            st.write("---")
            
    elif app_selection == "TMT Rod":
        st.header("TMT Rod Matching Between Buyer And Supplier")

        # Input fields for buyer
        st.sidebar.header("Enter Buyer Details")
        buyer_details = {
            "Price": st.sidebar.number_input("Price (min 50k )"),
            "Availability": st.sidebar.number_input("Availability (500 to 5000 Standard)"),
            "Grade": st.sidebar.number_input("Steel Rod Grade(W) (500 to 700)"),
            "Yeild_Strength": st.sidebar.number_input("Yield Strength (480 to 520 standard)"),
            "Tensile_Strength": st.sidebar.number_input("Tensile Strength (530 to 570 standard)"),
            "Elongation": st.sidebar.number_input("Elongation (12 to 16 standard)"),
            "Sulphar": st.sidebar.number_input("Sulphar % (0.03 to 0.7 standard)"),
            "Carbone": st.sidebar.number_input("Carbon % (0.175 to 0.245 standard)"),
            "Slicon": st.sidebar.number_input("Slicon % (0.155 to 0.345 standard)")
        }

        # Prediction button
        if st.sidebar.button("Search"):
            with st.spinner('Generating Results...'):
                predictions, suppliers, predicted_classes, predicted_probs = predict_match(
                    buyer_details["Price"],
                    buyer_details["Availability"],
                    buyer_details["Grade"],
                    buyer_details["Yeild_Strength"],
                    buyer_details["Tensile_Strength"],
                    buyer_details["Elongation"],
                    buyer_details["Sulphar"],
                    buyer_details["Carbone"],
                    buyer_details["Slicon"]
                )

            st.write("---")
            st.subheader("Dashboard:")
            for i, (prediction, supplier, predicted_class, predicted_prob) in enumerate(zip(predictions, suppliers, predicted_classes, predicted_probs), start=1):
                st.write(f"Search List {i}:")
                if predicted_class == 1:
                    # If predicted class is 1, there's a match
                    st.success(f"KPI Score: {predicted_prob[1]:0f}%")
                    st.subheader("Buyer Details:")
                    st.info("\n".join([f"{k}: {v}" for k, v in buyer_details.items()]))
                    st.subheader("Supplier Details:")
                    st.success("\n".join([f"Supplier_{k}: {v}" for k, v in supplier.items()]))
                    st.subheader("Overall KPI Match:")
                    st.success(f"Overall KPI Match between Buyer and Supplier: {predicted_prob[1]:.0f}%")
                else:
                    # If predicted class is 0, there's no match
                    no_match_prob = 1 - predicted_prob[0] * 100
                    st.error(f"No match found. Match Score: {no_match_prob:.0f}%")
                st.write("---")

if __name__ == "__main__":
    main()
