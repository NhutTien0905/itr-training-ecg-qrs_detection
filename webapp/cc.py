import os
import wfdb
import streamlit as st
import matplotlib.pyplot as plt

# Create a temporary directory to save uploaded files
temp_dir = "temp_ecg_data"
os.makedirs(temp_dir, exist_ok=True)

# Upload the .hea and .dat files
uploaded_hea_file = st.file_uploader("Upload the .hea file", type="hea")
uploaded_dat_file = st.file_uploader("Upload the .dat file", type="dat")

if uploaded_hea_file and uploaded_dat_file:
    # Save the uploaded files with unique names
    hea_file_path = os.path.join(temp_dir, uploaded_hea_file.name)
    dat_file_path = os.path.join(temp_dir, uploaded_dat_file.name)
    
    with open(hea_file_path, "wb") as f:
        f.write(uploaded_hea_file.getbuffer())

    with open(dat_file_path, "wb") as f:
        f.write(uploaded_dat_file.getbuffer())

    # Use the base name (without extensions) to read the record
    record_base_name = os.path.splitext(uploaded_hea_file.name)[0]
    record_path = os.path.join(temp_dir, record_base_name)
    
    # Read the record
    record = wfdb.rdrecord(record_path)
    
    # Get the signal
    signal = record.p_signal

    # Plot the signal
    fig, ax = plt.subplots()
    ax.plot(signal)
    st.pyplot(fig)
    
    # Display some metadata
    st.write(f"Record Name: {record.record_name}")
    st.write(f"Number of Signals: {record.n_sig}")
    st.write(f"Signal Names: {record.sig_name}")
    st.write(f"Sampling Frequency: {record.fs} Hz")
    st.write(f"Duration: {record.sig_len / record.fs} seconds")

    # Optionally, clean up the temporary directory
    # os.remove(hea_file_path)
    # os.remove(dat_file_path)
