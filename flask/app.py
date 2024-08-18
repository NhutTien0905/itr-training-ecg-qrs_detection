import streamlit as st
import altair as alt
import time
import os
from define import *
from dataloader_v2 import generate_data, predict_flask, process_uploaded_files
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit to wide mode
st.set_page_config(layout="wide", page_title='ECG Signal Viewer', page_icon=':heart:')

# define constant
path = MITDB_DIR
list_hea_files = [f for f in os.listdir(path) if f.endswith('.hea')]
# Create a list of options
options = list_hea_files


# Initialize session state
if 'idx_start' not in st.session_state:
    st.session_state['idx_start'] = 0

if 'log' not in st.session_state:
    st.session_state['log'] = ''

if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = '100.hea'

# load data
end = 360*3
# p_signal, atr_sample = load_ecg(path + st.session_state['selected_option'][:-4])

# Initialize session state for chart data, chart visibility, and chart updates
if 'update_chart' not in st.session_state:
    st.session_state['update_chart'] = False

if 'chart_visible' not in st.session_state:
    st.session_state['chart_visible'] = False

if 'forward' not in st.session_state:
    st.session_state['forward'] = False

if 'backward' not in st.session_state:
    st.session_state['backward'] = False


# Function to update log
def update_log(message):
    st.session_state['log'] += message + '\n'

# Callback functions
def option_callback():
    st.session_state['chart_visible'] = True
    # delete previous txt files
    path_txt = "tmp/"
    list_txt_files = [f for f in os.listdir(path_txt) if f.endswith('.txt')]
    for f in list_txt_files:
        os.remove(os.path.join(path_txt, f))
    update_log(f'Button 1 clicked: Loading data')

    # reset idx_start
    st.session_state['idx_start'] = 0
    # Stop chart updates when a new option is selected
    st.session_state['update_chart'] = False  

def button1_callback():
    # if st.session_state['selected_option'] is not None:
    st.session_state['chart_visible'] = True
    
    update_log('Button 2 clicked: Predicting...')
    predict_flask(p_signal)

    st.session_state['update_chart'] = False

def button2_callback():
    st.session_state['update_chart'] = not st.session_state['update_chart']  # Toggle chart updates
    if st.session_state['update_chart']:
        update_log('Button 3 clicked: Starting updates')
    else:
        update_log('Button 3 clicked: Stopping updates')

def button_forward_callback():
    st.session_state['idx_start'] += end  # Increment idx_start by 360*time_input
    if st.session_state['idx_start'] >= len(p_signal) - end:
        st.session_state['idx_start'] = len(p_signal) - end

def button_backward_callback():
    st.session_state['idx_start'] -= end  # Decrement idx_start by 360*time_input
    if st.session_state['idx_start'] < 0:
        st.session_state['idx_start'] = 0

# Create two columns with custom width ratios
col1, col2 = st.columns([1, 4])

# Column 1
with col1:
    # Create a temporary directory to save uploaded files
    temp_dir = "temp_ecg_data"
    os.makedirs(temp_dir, exist_ok=True)

    # Upload the .hea and .dat files
    uploaded_hea_file = st.file_uploader("Upload the .hea file", type="hea")
    uploaded_dat_file = st.file_uploader("Upload the .dat file", type="dat")
    
    if uploaded_hea_file and uploaded_dat_file:
        p_signal = process_uploaded_files(uploaded_hea_file, uploaded_dat_file)
    else:
        st.session_state['chart_visible'] = False
        try:
            # delete previous uploaded files
            path_tmp = "temp_ecg_data/"
            list_data_files = [f for f in os.listdir(path_tmp) if f.endswith('.hea') or f.endswith('.dat')]
            for f in list_data_files:
                os.remove(os.path.join(path_tmp, f))
        except:
            pass

    # Create a row for buttons
    button_col1_1, button_col1, button_col2 = st.columns(3)
    with button_col1_1:
        st.button('Load data', on_click=option_callback)#
    with button_col1:
        st.button('Predict', on_click=button1_callback)
    with button_col2:
        st.button('Start/Stop showing', on_click=button2_callback)
    
    # Display log messages in a text area
    st.text_area('Log', st.session_state['log'], height=200)

# Column 2
with col2:
    # Centering the Predict1 and Predict2 buttons in the same row
    _, button_col3, button_col4, input_text, _ = st.columns([1, 1, 1, 1, 1])
    with button_col3:
        st.button(':arrow_backward:', on_click=button_backward_callback)
    with button_col4:
        st.button(':arrow_forward:', on_click=button_forward_callback)
    with input_text:
        time_input = st.text_input('Time display (s)', value='3')

    # Update the 'end' variable based on the text input
    try:
        end = 360 * int(time_input)
    except ValueError:
        st.error("Please enter a valid integer for time display.")
        end = 360*3  # Default value if the input is invalid

    if st.session_state['chart_visible']:
        # Get the generated data
        data = generate_data(p_signal, st.session_state['idx_start'], end)

        # Define colors for each series
        color_scale = alt.Scale(domain=['Signal', 'Predicted_QRS'], range=['red', 'green'])
        color_scale1 = alt.Scale(domain=['Signal1', 'Predicted_QRS1'], range=['red', 'green'])

        # Determine the total time span of your data
        total_time = data['x'].max() - data['x'].min()
        pixels_per_interval = 10  # Number of pixels per 0.04 second interval

        # Calculate the width of the chart
        chart_width = (total_time / 0.04) * pixels_per_interval

        # Create the list of ticks at 0.04s intervals
        tick_values = [round(data['x'].min() + i * 0.04, 2) for i in range(int(total_time / 0.04) + 1)]

        # Determine the total time span of your data
        total_range = data['value'].max() - data['value'].min()

        # Calculate the width of the chart
        chart_height = 35 * pixels_per_interval

        # Create the list of ticks at 0.04s intervals
        tick_values_y = [round(data['value'].min() + i * 0.1, 2) for i in range(int(total_range / 0.1) + 1)]

        # Create an Altair line chart for Signal channel 0
        line_chart = alt.Chart(data[data['series'] == 'Signal']).mark_line().encode(
            x=alt.X('x', axis=alt.Axis(title='Time (s)', values=tick_values, grid = True)),
            y=alt.Y('value', axis=alt.Axis(title='Voltage (mV)', values=tick_values_y, grid = True)),
            color=alt.Color('series', scale=color_scale, legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        ).properties(
            title='ECG signal channel 0',
            width=chart_width,  # Set the overall width of the chart
            height=chart_height  # Set the height of the chart
        )

        # Create an Altair scatter chart for predicted QRS
        try:
            scatter_chart_pred = alt.Chart(data[data['series'] == 'Predicted_QRS']).mark_point().encode(
                x=alt.X('x', axis=alt.Axis(title='Time (s)', values=tick_values, grid = True)),
                y=alt.Y('value', axis=alt.Axis(title='Voltage (mV)', values=tick_values_y, grid = True)),
                color=alt.Color('series', scale=color_scale, legend=alt.Legend(title="Series")),  # Use series for color and legend
                tooltip=['x', 'value']  # Adding tooltip for interactive display
            ).properties(
                # title='ECG signal channel 0',
                width=chart_width,  # Set the overall width of the chart
                height=chart_height  # Set the height of the chart
            )

            # Combine all charts
            combined_chart = alt.layer(line_chart, scatter_chart_pred).resolve_scale(
                y='shared'
            )
        except:
            # Combine line chart and real QRS scatter chart
            combined_chart = alt.layer(line_chart).resolve_scale(
                y='shared'
            )

        # Create an Altair line chart for Signal channel 1
        line_chart1 = alt.Chart(data[data['series'] == 'Signal1']).mark_line().encode(
            x=alt.X('x', axis=alt.Axis(title='Time (s)', values=tick_values, grid = True)),
            y=alt.Y('value', axis=alt.Axis(title='Voltage (mV)', values=tick_values_y, grid = True)),
            color=alt.Color('series', scale=color_scale1, legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        ).properties(
            title='ECG signal channel 1',
            width=chart_width,  # Set the overall width of the chart
            height=chart_height  # Set the height of the chart
        )

        try:
            # Create an Altair scatter chart for predicted QRS
            scatter_chart_pred1 = alt.Chart(data[data['series'] == 'Predicted_QRS1']).mark_point().encode(
                x=alt.X('x', axis=alt.Axis(title='Time (s)', values=tick_values, grid = True)),
                y=alt.Y('value', axis=alt.Axis(title='Voltage (mV)', values=tick_values_y, grid = True)),
                color=alt.Color('series', scale=color_scale1, legend=alt.Legend(title="Series")),  # Use series for color and legend
                tooltip=['x', 'value']  # Adding tooltip for interactive display
            ).properties(
                # title='ECG signal channel 0',
                width=chart_width,  # Set the overall width of the chart
                height=chart_height  # Set the height of the chart
            )

            # Combine all charts
            combined_chart1 = alt.layer(line_chart1, scatter_chart_pred1).resolve_scale(
                y='shared'
            )
        except:
            # Combine line chart and real QRS scatter chart
            combined_chart1 = alt.layer(line_chart1).resolve_scale(
                y='shared'
            )

        st.altair_chart(combined_chart, use_container_width=True)
        st.altair_chart(combined_chart1, use_container_width=True)

# Automatic chart updates every 100ms if update_chart is True
if st.session_state['update_chart']:
    time.sleep(0.1)
    st.session_state['idx_start'] += 1  # Increment idx_start
    st.session_state['chart_data'] = generate_data(p_signal, st.session_state['idx_start'], end)
    st.experimental_rerun()

if st.session_state['forward'] or st.session_state['backward']:
    # st.session_state['idx_start'] += end  # Increment idx_start by 360*time_input
    st.session_state['chart_data'] = generate_data(p_signal, st.session_state['idx_start'], end)
    st.experimental_rerun()