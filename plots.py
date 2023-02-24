from statsmodels.graphics.tsaplots import acf, pacf
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_pacf(series:pd.Series, series_name:str, nlags:int, partial:bool=True, save:bool=False):
    
    '''
    This func estimates the (partial) autocorrelation function of a given time series using statsmodels package and then plots it using plotly package
    Note that the default pacf calculation method is Yule-Walker with sample size-adjustment in denominator for acovf
    Note that the default confidence interval is 95%
    
    :param series: observations of time series for which acf/pacf is calculated
    :type series: pd.Series
    :param series_name: at least the name and the frequency of the time series, without special characters, that will be used to create a title
    :type series_name: str
    :param nlags: number of lags to return autocorrelation for
    :type nlags: int
    :param partial: whether to plot pacf
    :type partial: bool (default True)
    :param save: whether to save the plot to the current workding directory
    :type save: bool (default False)
    '''
    
    if not isinstance(series, pd.Series):
        raise Exception('Time series is not a pd.Series type!')

    # Define the title depending on the bool argument
    title=f'PACF of {series_name}' if partial else f'ACF of {series_name}'
    
    # Calculate the acf/pacf and the confidence intervals
    corr_array, conf_int_array = pacf(series.dropna(), alpha=0.05, nlags=nlags, method='yw') if partial else acf(series.dropna(), alpha=0.05, nlags=nlags)
    
    # Center the confidence intervals so that it's easy to visually inspect if a given correlation is significantly different from zero
    lower_y = conf_int_array[:,0] - corr_array
    upper_y = conf_int_array[:,1] - corr_array
    
    # Create an empty figure
    fig = go.Figure()

    # Plot the correlations using vertical lines
    [fig.add_scatter(x=(x,x), y=(0,corr_array[x]), mode='lines', line_color='#3f3f3f', hoverinfo='skip') 
        for x in np.arange(len(corr_array))]
    
    # Plot the correlations using markers
    # The <extra></extra> part removes the trace name
    fig.add_scatter(
        x=np.arange(len(corr_array)),
        y=corr_array,
        mode='markers',
        marker_color='#1f77b4',
        marker_size=12,
        hovertemplate=
        'Lag %{x}<br>' +
        'Corr: %{y:.2f}<br>' +
        '<extra></extra>'
    )
    
    # Plot the centered confidence intervals
    fig.add_scatter(x=np.arange(len(corr_array)), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)', hoverinfo='skip')
    fig.add_scatter(x=np.arange(len(corr_array)), y=lower_y, mode='lines', fillcolor='rgba(32, 146, 230,0.3)',
        fill='tonexty', line_color='rgba(255,255,255,0)', hoverinfo='skip')
    
    # Prettify the plot
    fig.update_traces(showlegend=False)
    fig.update_xaxes(tickvals=np.arange(start=0, stop=nlags+1))
    fig.update_yaxes(zerolinecolor='#000000')
    fig.update_layout(title=title, title_x=0.5, width=500, height=300, margin=dict(l=0, r=0, b=0, t=30, pad=1))
    # fig.update_layout(title=title, title_x=0.5, width=500, height=300, hovermode=False, margin=dict(l=0, r=0, b=0, t=30, pad=1))

    # Save the plot to the current working directory
    if save:
        fig.write_image(f'''{title.replace(' ', '_')}.png''')

    # Eventually show the plot
    fig.show()