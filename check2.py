import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import base64
import numpy as np
df = px.data.iris()

@st.cache_data 
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/free-vector/geometric-shapes-border-vector-background-design-space_53876-177956.jpg?t=st=1716367594~exp=1716371194~hmac=0fce2b34bb72931fba4992bf79ec857bfcf94f891664eae2453549795692c722&w=740");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .big-font {
        font-size:50px !important;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
def normalize(matrix):
    mean = np.mean(matrix, axis=0)
    std_dev = np.std(matrix, axis=0)
    normalized_matrix = (matrix - mean) / std_dev
    return normalized_matrix, mean, std_dev

def denormalize(normalized_matrix, mean, std_dev):
    denormalized_matrix = (normalized_matrix * std_dev) + mean
    return denormalized_matrix

def calculate_matrices(D, P, R, K, regularization=1e-10):
    D_normalized, D_mean, D_std_dev = normalize(D)
    P_normalized, P_mean, P_std_dev = normalize(P)
    R_normalized, R_mean, R_std_dev = normalize(R)
    K_normalized, K_mean, K_std_dev = normalize(K)

    # Compute matrix C
    C = np.dot(D_normalized, np.dot(P_normalized, R_normalized))
    
    # Add regularization to the diagonal of C
    U, S, VT = np.linalg.svd(C, full_matrices=False)
    S = np.diag(S)
    S_inv = np.diag(1. / (S.diagonal() + regularization))
    C_pseudo_inv = np.dot(VT.T, np.dot(S_inv, U.T))
    
    # Compute the weight matrix W
    W = np.dot(C_pseudo_inv, K_normalized)
    
    # Verification
    verification = np.dot(np.dot(D_normalized, np.dot(P_normalized, R_normalized)), W)
    verification_denormalized = denormalize(verification, K_mean, K_std_dev)
    
    errors = []
    for i in range(len(verification_denormalized)):
        row_errors = []
        for j in range(len(verification_denormalized[0])):
            if K[i][j] != 0:
                error = round(abs((K[i][j] - verification_denormalized[i][j])) / K[i][j], 2)
            else:
                error = np.nan
            row_errors.append(error)
        errors.append(row_errors)
    
    errors = np.array(errors)
    
    return W, verification_denormalized, errors


def display_matrix(matrix, title):
    st.write(f"## {title}")
    st.dataframe(pd.DataFrame(matrix))

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'a' not in st.session_state:
    st.session_state.a = 0
if 'b' not in st.session_state:
    st.session_state.b = 0
if 'c' not in st.session_state:
    st.session_state.c = 0
if 'd' not in st.session_state:
    st.session_state.d = 0
if 'e' not in st.session_state:
    st.session_state.e = 0
if 'D' not in st.session_state:
    st.session_state.D = None
if 'P' not in st.session_state:
    st.session_state.P = None
if 'R' not in st.session_state:
    st.session_state.R = None
if 'K' not in st.session_state:
    st.session_state.K = None
if 'W' not in st.session_state:
    st.session_state.W = None
if 'verification_denormalized' not in st.session_state:
    st.session_state.verification_denormalized = None
if 'errors' not in st.session_state:
    st.session_state.errors = None

# Navigation
def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1
def curr_page():
    st.session_state.page+=0
# Page 0
if st.session_state.page == 0:
    st.title("Welcome to Coke Quality Predictor")
    
    
    st.subheader("Enter your details below:")
    name = st.text_input("Enter your Name")
    id = st.text_input("Enter your ID")
    
    if st.button("Submit"):
        st.session_state.name = name
        st.session_state.id = id
        next_page()  # Move to the next page    

# Page 1: User Inputs
elif st.session_state.page == 1:
    st.title("Enter Initial Parameters")

    a = st.number_input("Enter Number of days", min_value=1)
    b = st.number_input("Enter Number of Sources", min_value=1)
    
    sources = ["KEDLA", "KATHARA", "BELATAND", "MADHUBAN", "CHASNALA", "PATHARDIH", "JHAMADOBA",
               "AUST. HARD", "AUST. LVHR", "USA HARD", "BENGA HARD", "APPIN HARD", "TUHUP HARD",
               "AUST. SOFT", "UAS SOFT", "RUSSIAN SOFT"]
    
    if b:
        selected_sources = st.multiselect("Select Sources", sources, max_selections=b)
    
    c = st.number_input("Enter Number of Properties", min_value=1)
    
    properties = ["Ash", "VM", "Moisture", "FC", "Bulk Density", "Crushing index", "Crushing index (0.5 mm)",
                  "MMR", "Fluidity", "CSN / FSI", "Softening temperature", "Resolidification temperature", 
                  "Sulpher", "Phosporous"]
    
    if c:
        selected_properties = st.multiselect("Select Properties", properties, max_selections=c)
    
    d = st.number_input("Enter Number of Process Parameters", min_value=1)
    
    process_parameters = ["Charging tonnage (Dry)", "Moisture content", "Bulk Density of charging coal",
                          "Average Charging temperature (P/S)", "Average Charging temperature (C/S)", 
                          "Coke Mass Temperature in degC", "Cross wall temperature", "Push Force / Pushing Current (Min)", 
                          "Push Force / Pushing Current (Max)", "PRI ( Pushing regularity Index)", "Coke per push in dry basis", 
                          "Gross coke Yield", "GCM Pressure", "GCM Temperature", "Coking time", "Coke end temperature", 
                          "Quenching time", "Quenching water volume", "GCV of mixed gas", "GCV of BF Gas", 
                          "Underfiring CO Gas Qty / day", "Underfiring BF Gas Qty / day"]
    
    if d:
        selected_process_parameters = st.multiselect("Select Process Parameters", process_parameters, max_selections=d)
    
    e = st.number_input("Enter Number of Output Coke Properties", min_value=1)
    
    output_coke_properties = ["Ash", "VM", "Moisture", "FC", "AMS", "<25 mm", ">80mm", "CSR", "CRI", "M40", "M10", "S", "P"]
    
    if e:
        selected_output_coke_properties = st.multiselect("Select Output Coke Properties", output_coke_properties, max_selections=e)

    if st.button("Next Page1"):
        st.session_state.a = a
        st.session_state.b = b
        st.session_state.c = c
        st.session_state.d = d
        st.session_state.e = e
        st.session_state.selected_sources = selected_sources
        st.session_state.selected_properties = selected_properties
        st.session_state.selected_process_parameters = selected_process_parameters
        st.session_state.selected_output_coke_properties = selected_output_coke_properties
        next_page()
    
    if st.button("Previous Page1"):
        prev_page()    

# Page 2: Input Coal Percentages Matrix (D)
elif st.session_state.page == 2:
    st.title("Enter Percentages of Input Coal from Different Sources")
    
    D = np.zeros((st.session_state.a, st.session_state.b))
    for i in range(st.session_state.a):
        cols = st.columns(st.session_state.b + 1)  # Create columns for inputs + day label
        cols[0].write(f"Day {i+1}")
        for j in range(st.session_state.b):
            D[i, j] = cols[j + 1].number_input(f"{st.session_state.selected_sources[j]}", min_value=0.0, step=0.01, key=f"D_{i}_{j}")

    if st.button("Previous page2"):
        st.session_state.D = D
        prev_page()
    if st.button("Next page2"):
        st.session_state.D = D
        next_page()
    # if st.button("Next"):
    #     next_page()    
# elif st.session_state.page == 3:
    # next_page()
# Page 3: Input Coal Properties Matrix (P)
elif st.session_state.page == 3:
    st.title("Enter Properties of Individual Coal from Different Sources")
    
    P = np.zeros((st.session_state.b, st.session_state.c))
    for i in range(st.session_state.b):
        cols = st.columns(st.session_state.c + 1)  # Create columns for inputs + source label
        cols[0].write(st.session_state.selected_sources[i])
        for j in range(st.session_state.c):
            P[i, j] = cols[j + 1].number_input(f"{st.session_state.selected_properties[j]}", min_value=0.0, step=0.01, key=f"P_{i}_{j}")

    if st.button("Previous page3"):
        st.session_state.P = P
        prev_page()
    if st.button("Next page3"):
        st.session_state.P = P
        next_page()
# elif st.session_state.page == 4:
#     next_page()
# Page 4: Input Process Parameters Matrix (R)
elif st.session_state.page == 4:
    st.title("Enter Process Parameters")
    
    R = np.zeros((st.session_state.a, st.session_state.d))
    for i in range(st.session_state.a):
        cols = st.columns(st.session_state.d + 1)  # Create columns for inputs + day label
        cols[0].write(f"Day {i+1}")
        for j in range(st.session_state.d):
            R[i, j] = cols[j + 1].number_input(f"{st.session_state.selected_process_parameters[j]}", min_value=0.0, step=0.01, key=f"R_{i}_{j}")

    if st.button("Previous Page4"):
        st.session_state.R = R
        prev_page()
    if st.button("Next Page4"):
        st.session_state.R = R
        next_page()
    # if st.button("Next"):
    #     st.session_state.R = R
    #     next_page() 
# elif st.session_state.page == 6:
#     next_page()
# Page 5: Input Coke Properties Matrix (K)
elif st.session_state.page == 5:
    st.title("Enter Properties of Output Coke")

    K = np.zeros((st.session_state.a, st.session_state.e))
    for i in range(st.session_state.a):
        cols = st.columns(st.session_state.e + 1)  # Create columns for inputs + day label
        cols[0].write(f"Day {i+1}")
        for j in range(st.session_state.e):
            K[i, j] = cols[j + 1].number_input(f"{st.session_state.selected_output_coke_properties[j]}", min_value=0.0, step=0.01, key=f"K_{i}_{j}")

    if st.button("Previous page5"):
        st.session_state.K = K
        prev_page()
    if st.button("Next page5"):
        st.session_state.K = K
        next_page()
# elif st.session_state.page == 8:
#     next_page()
# Page 6: Calculate Matrices and Show Results
elif st.session_state.page == 6:
    st.title("Calculation and Results")

    D = st.session_state.D
    P = st.session_state.P
    R = st.session_state.R
    K = st.session_state.K

    if D is None or P is None or R is None or K is None:
        st.error("One of the matrices is not initialized. Please check your inputs.")
    else:
        W, verification_denormalized, errors = calculate_matrices(D, P, R, K)

        st.session_state.W = W
        st.session_state.verification_denormalized = verification_denormalized
        st.session_state.errors = errors

    if st.button("Previous Page6"):
        
        curr_page()
    if st.button("Show Results"):
        
        next_page()

# Page 7: Display Weight Matrix
elif st.session_state.page == 7:
    st.title("Weight Matrix")
    display_matrix(st.session_state.W, "Weight Matrix")

    if st.button("Previous page7"):
        
        prev_page()
    if st.button("Next page7"):
        
        next_page()

# Page 8: Display Verification Results
elif st.session_state.page == 8:
    st.title("Verification Results")
    display_matrix(st.session_state.verification_denormalized, "Verification Results")

    if st.button("Previous Page8"):
        
        prev_page()
    if st.button("Next Page8"):
        
        next_page()

# Page 9: Display Error Matrix
elif st.session_state.page == 9:
    st.title("Error Matrix")
    display_matrix(st.session_state.errors, "Error Matrix")

    if st.button("Previous page9"):
        
        prev_page()
    if st.button("Next page9"):
        
        next_page()

# Page 10: Thank You
elif st.session_state.page == 10:
    st.title("Thank You")
    st.write("Thank you for using the application!")

    if st.button("Previous Page10"):
        prev_page()
