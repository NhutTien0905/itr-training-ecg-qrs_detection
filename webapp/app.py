import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
import os
import wfdb
from define import *
from detectors import QRSDetector
from preprocessing import preprocess_data

# Set Streamlit to wide mode
st.set_page_config(layout="wide", page_title='ECG Signal Viewer', page_icon=':heart:')

# define constant
path = MITDB_DIR
list_hea_files = [f for f in os.listdir(path) if f.endswith('.hea')]
# Create a list of options
options = list_hea_files

# function for load data
def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file, 'atr')
    p_signal = record.p_signal
    atr_sample = annotation.sample
   
    return p_signal, atr_sample

# Sample data generation for the line chart and scatter chart with different x-values
def generate_data(p_signal, atr_sample, start, end):
    # Signal
    bias = 35
    x_A = np.arange(start, start + end)/360
    value_A = p_signal[start + bias:bias + start + end, 0]

    # QRS
    x_B = [(i-bias)/360 for i in atr_sample if start < i < start + end and i > 2*bias]
    value_B = p_signal[[i for i in atr_sample if start < i < start + end and i > 2*bias],0]

    # Predicted QRS
    detector = QRSDetector(preprocess_data(value_A))
    qrs = detector.detect_qrs()
    classes = np.argmax(qrs, axis=1)
    idx_qrs = np.where(classes == 1)[0] + bias
    x_C = [(i+start) / 360 for i in idx_qrs]
    value_C = value_A[idx_qrs]

    signal_data = pd.DataFrame({
        'x': x_A,
        'value': value_A,
        'series': 'Signal'
    })
    qrs_data = pd.DataFrame({
        'x': x_B,
        'value': value_B,
        'series': 'QRS'
    })
    pred_qrs = pd.DataFrame({
        'x': x_C,
        'value': value_C,
        'series': 'Predicted_QRS'
    })
    return pd.concat([signal_data, qrs_data, pred_qrs])


# Initialize session state
if 'idx_start' not in st.session_state:
    st.session_state['idx_start'] = 0

if 'log' not in st.session_state:
    st.session_state['log'] = ''

if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = '100.hea'

# load data
end = 145*4
p_signal, atr_sample = load_ecg(path + st.session_state['selected_option'][:-4])

# Initialize session state for chart data, chart visibility, and chart updates
if 'update_chart' not in st.session_state:
    st.session_state['update_chart'] = False

if 'chart_data' not in st.session_state:
    st.session_state['chart_data'] = generate_data(p_signal, atr_sample, st.session_state['idx_start'], end)

if 'chart_visible' not in st.session_state:
    st.session_state['chart_visible'] = False



# Function to update log
def update_log(message):
    st.session_state['log'] += message + '\n'

# Callback functions
def option_callback():
    selected_option = st.session_state.selected_option
    if selected_option is not None:
        st.session_state['chart_visible'] = True
        update_log('Selected Option 1: Showing data')

    st.session_state['update_chart'] = False  # Stop chart updates when a new option is selected

def button1_callback():
    if st.session_state['selected_option'] is not None:
        st.session_state['chart_data'] = generate_data(p_signal, atr_sample, st.session_state['idx_start'], end)
        st.session_state['chart_visible'] = True
        update_log('Button 1 clicked: Showing data')

    st.session_state['update_chart'] = False

def button2_callback():
    st.session_state['update_chart'] = not st.session_state['update_chart']  # Toggle chart updates
    if st.session_state['update_chart']:
        update_log('Button 2 clicked: Starting updates')
    else:
        update_log('Button 2 clicked: Stopping updates')


# Create two columns with custom width ratios
col1, col2 = st.columns([1, 4])

# Column 1
with col1:
    # Selectbox with a callback
    st.selectbox(
        'Select an option:',
        options,
        index=options.index(st.session_state.selected_option),
        key='selected_option',
        on_change=option_callback
    )
    
    # Create a row for buttons
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        st.button('Load data', on_click=button1_callback)
    with button_col2:
        st.button('Start', on_click=button2_callback)
    
    # Display log messages in a text area
    st.text_area('Log', st.session_state['log'], height=200)

# Column 2
with col2:
    if st.session_state['chart_visible']:
        # Get the generated data
        data = st.session_state['chart_data']

        # Create an Altair line chart for Signal
        line_chart = alt.Chart(data[data['series'] == 'Signal']).mark_line().encode(
            # x='x',
            x=alt.X('x', axis=alt.Axis(title='Time (s)')),
            y='value',
            color=alt.Color('series', legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        ).properties(
            title='ECG signal'
        )

        # Create an Altair scatter chart for QRS
        scatter_chart_real = alt.Chart(data[data['series'] == 'QRS']).mark_point().encode(
            # x='x',
            x=alt.X('x', axis=alt.Axis(title='Time (s)')),
            y='value',
            color=alt.Color('series', legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        )

        # Create an Altair scatter chart for preidcted QRS
        scatter_chart_pred = alt.Chart(data[data['series'] == 'Predicted_QRS']).mark_point().encode(
            # x='x',
            x=alt.X('x', axis=alt.Axis(title='Time (s)')),
            y='value',
            color=alt.Color('series', legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        )

        # Combine both charts
        combined_chart = alt.layer(line_chart, scatter_chart_real, scatter_chart_pred).resolve_scale(
            y='shared'
        )

        st.altair_chart(combined_chart, use_container_width=True)

# Automatic chart updates every 100ms if update_chart is True
if st.session_state['update_chart']:
    time.sleep(0.05)
    st.session_state['idx_start'] += 1  # Increment idx_start
    st.session_state['chart_data'] = generate_data(p_signal, atr_sample, st.session_state['idx_start'], end)
    st.experimental_rerun()