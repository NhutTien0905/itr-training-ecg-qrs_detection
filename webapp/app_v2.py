import streamlit as st
import altair as alt
import time
import os
from define import *
from dataloader import load_ecg, generate_data, predict
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
end = 360*10
p_signal, atr_sample = load_ecg(path + st.session_state['selected_option'][:-4])

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
    selected_option = st.session_state.selected_option
    if selected_option is not None:
        st.session_state['chart_visible'] = True
        update_log(f'Selected option: {selected_option}')
    
    # delete previous txt files
    path_txt = "tmp/"
    list_txt_files = [f for f in os.listdir(path_txt) if f.endswith('.txt')]
    for f in list_txt_files:
        os.remove(os.path.join(path_txt, f))

    # reset idx_start
    st.session_state['idx_start'] = 0
    # Stop chart updates when a new option is selected
    st.session_state['update_chart'] = False  

def button1_callback():
    if st.session_state['selected_option'] is not None:
        st.session_state['chart_visible'] = True
        update_log('Button 1 clicked: Predicting...')
        predict(p_signal)

    st.session_state['update_chart'] = False

def button2_callback():
    st.session_state['update_chart'] = not st.session_state['update_chart']  # Toggle chart updates
    if st.session_state['update_chart']:
        update_log('Button 2 clicked: Starting updates')
    else:
        update_log('Button 2 clicked: Stopping updates')

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
        st.button('Predict', on_click=button1_callback)
    with button_col2:
        st.button('Start/Stop showing', on_click=button2_callback)
    
    # Display log messages in a text area
    st.text_area('Log', st.session_state['log'], height=500)

# Column 2
with col2:
    # Centering the Predict1 and Predict2 buttons in the same row
    _, button_col3, button_col4, input_text, _ = st.columns([1, 1, 1, 1, 1])
    with button_col3:
        st.button(':arrow_backward:', on_click=button_backward_callback)
    with button_col4:
        st.button(':arrow_forward:', on_click=button_forward_callback)
    with input_text:
        time_input = st.text_input('Time display (s)', value='10')

    # Update the 'end' variable based on the text input
    try:
        end = 360 * int(time_input)
    except ValueError:
        st.error("Please enter a valid integer for time display.")
        end = 3600  # Default value if the input is invalid

    if st.session_state['chart_visible']:
        # Get the generated data
        data = generate_data(p_signal, atr_sample, st.session_state['idx_start'], end)

        # Define colors for each series
        color_scale = alt.Scale(domain=['Signal', 'QRS', 'Predicted_QRS'], range=['red', 'green', 'blue'])
        color_scale1 = alt.Scale(domain=['Signal1', 'QRS1', 'Predicted_QRS1'], range=['red', 'green', 'blue'])

        # Create an Altair line chart for Signal channel 0
        line_chart = alt.Chart(data[data['series'] == 'Signal']).mark_line().encode(
            x=alt.X('x', axis=alt.Axis(title='Time (s)')),
            y='value',
            color=alt.Color('series', scale=color_scale, legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        ).properties(
            title='ECG signal channel 0'
        )

        # Create an Altair scatter chart for QRS
        scatter_chart_real = alt.Chart(data[data['series'] == 'QRS']).mark_point().encode(
            x=alt.X('x', axis=alt.Axis(title='Time (s)')),
            y='value',
            color=alt.Color('series', scale=color_scale, legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        )

        # Create an Altair scatter chart for predicted QRS
        try:
            scatter_chart_pred = alt.Chart(data[data['series'] == 'Predicted_QRS']).mark_point().encode(
                x=alt.X('x', axis=alt.Axis(title='Time (s)')),
                y='value',
                color=alt.Color('series', scale=color_scale, legend=alt.Legend(title="Series")),  # Use series for color and legend
                tooltip=['x', 'value']  # Adding tooltip for interactive display
            )
            # Combine all charts
            combined_chart = alt.layer(line_chart, scatter_chart_real, scatter_chart_pred).resolve_scale(
                y='shared'
            )
        except:
            # Combine line chart and real QRS scatter chart
            combined_chart = alt.layer(line_chart, scatter_chart_real).resolve_scale(
                y='shared'
            )

        # Create an Altair line chart for Signal channel 1
        line_chart1 = alt.Chart(data[data['series'] == 'Signal1']).mark_line().encode(
            x=alt.X('x', axis=alt.Axis(title='Time (s)')),
            y='value',
            color=alt.Color('series', scale=color_scale1, legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        ).properties(
            title='ECG signal channel 1'
        )

        # Create an Altair scatter chart for QRS
        scatter_chart_real1 = alt.Chart(data[data['series'] == 'QRS1']).mark_point().encode(
            x=alt.X('x', axis=alt.Axis(title='Time (s)')),
            y='value',
            color=alt.Color('series', scale=color_scale1, legend=alt.Legend(title="Series")),  # Use series for color and legend
            tooltip=['x', 'value']  # Adding tooltip for interactive display
        )

        try:
            # Create an Altair scatter chart for predicted QRS
            scatter_chart_pred1 = alt.Chart(data[data['series'] == 'Predicted_QRS1']).mark_point().encode(
                x=alt.X('x', axis=alt.Axis(title='Time (s)')),
                y='value',
                color=alt.Color('series', scale=color_scale1, legend=alt.Legend(title="Series")),  # Use series for color and legend
                tooltip=['x', 'value']  # Adding tooltip for interactive display
            )

            # Combine all charts
            combined_chart1 = alt.layer(line_chart1, scatter_chart_real1, scatter_chart_pred1).resolve_scale(
                y='shared'
            )
        except:
            # Combine line chart and real QRS scatter chart
            combined_chart1 = alt.layer(line_chart1, scatter_chart_real1).resolve_scale(
                y='shared'
            )

        st.altair_chart(combined_chart, use_container_width=True)
        st.altair_chart(combined_chart1, use_container_width=True)

# Automatic chart updates every 100ms if update_chart is True
if st.session_state['update_chart']:
    time.sleep(0.1)
    st.session_state['idx_start'] += 1  # Increment idx_start
    st.session_state['chart_data'] = generate_data(p_signal, atr_sample, st.session_state['idx_start'], end)
    st.experimental_rerun()

if st.session_state['forward'] or st.session_state['backward']:
    # st.session_state['idx_start'] += end  # Increment idx_start by 360*time_input
    st.session_state['chart_data'] = generate_data(p_signal, atr_sample, st.session_state['idx_start'], end)
    st.experimental_rerun()