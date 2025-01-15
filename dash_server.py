import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the dataset
data = pd.read_csv('./mental_health_dataset/1- mental-illnesses-prevalence.csv')

# Extract the GDP data (Assuming we have another dataframe for GDP)
df = pd.read_csv('./gdp/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2.csv')

years = [str(year) for year in range(1990, 2020)]  # Create a list of years as strings
columns_to_keep = ['Country Code'] + years  # Keep 'Country Code' + the years between 1990 and 2019

# Filter the dataframe to only include the necessary columns
filtered_df = df[columns_to_keep]
gdp_df_normalized=filtered_df
gdp_df_normalized.iloc[:, 1:] = gdp_df_normalized.iloc[:, 1:].div(gdp_df_normalized['1990'], axis=0)
gdp_df_filtered = gdp_df_normalized[gdp_df_normalized['1990'].notna()]


gdp_country_codes = gdp_df_filtered['Country Code'].unique()
health_country_codes = data['Code'].unique()


list_of_countries = list(set(gdp_country_codes) & set(health_country_codes))


# List of available metrics
metrics = [
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized',
    'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized',
    'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized',
    'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized',
    'Eating disorders (share of population) - Sex: Both - Age: Age-standardized'
]

# Get the unique years sorted
unique_years = sorted(data["Year"].unique())

# Normalize the metrics
def normalize_metric(data, metric):
    first_year_values = data.groupby('Entity').apply(lambda x: x[x['Year'] == 1990][metric].iloc[0])
    norm_column = 'Normalized ' + metric
    data[norm_column] = data.apply(lambda row: row[metric] / first_year_values[row['Entity']], axis=1)
    return norm_column

normalized_columns = {}
for metric in metrics:
    normalized_columns[metric] = normalize_metric(data, metric)


mental_health_df_filtered_by_country = data[data['Code'].isin(list_of_countries)]

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True )

# Default layout for the app
app.layout = html.Div([
    html.Div(id="page-content"),  # Placeholder for dynamic content
    html.Button("Previous", id="previous-page-button", n_clicks=0, style={
        "position": "fixed",  # Position fixed so they stay in view even when scrolling
        "bottom": "10px",  # Distance from the bottom of the page
        "left": "10px",  # Distance from the left side of the page
        "zIndex": "1000"  # Ensure the buttons are on top of other elements
    }),
    html.Button("Next", id="next-page-button", n_clicks=0, style={
        "position": "fixed",  # Position fixed so they stay in view even when scrolling
        "bottom": "10px",  # Distance from the bottom of the page
        "right": "10px",  # Distance from the right side of the page
        "zIndex": "1000"  # Ensure the buttons are on top of other elements
    }),

    
    dcc.Store(id="current-page", data=0)  # Track current page index
])

# Page 0 (World/Europe Visualization)
def render_page_0(selected_metric, toggle_europe):
    target_column = normalized_columns[selected_metric]

    # Default to the first year for initialization
    current_year = unique_years[0]
    filtered_data = data[data["Year"] == current_year]

    # Create the choropleth map
    fig = px.choropleth(
        filtered_data,
        locations="Code",  # ISO-3 codes
        locationmode="ISO-3",  # Specify ISO-3 format
        color=target_column,
        color_continuous_scale="Viridis",
        range_color=[
            data[target_column].min(),
            data[target_column].max()
        ],
        title=f"{selected_metric} in {current_year}"
    )
    fig.update_layout(
        geo=dict(
            scope='europe' if toggle_europe else 'world',  # Toggle between Europe and World view
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="white",
            projection_type="natural earth"
        ),
        width=1720,
        height=990,
        coloraxis_colorbar=dict(
            title="Normalized wrt 1990",
            thickness=20
        )
    )

    return html.Div([
        html.H1("Continuous World Map Evolution by Year"),
        html.Div([
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[{"label": metric, "value": metric} for metric in metrics],
                value=selected_metric,  # Default selection
                style={"width": "70%"}
            ),
            html.Button(
                "Zoom on Europe" if not toggle_europe else "Zoom on World",
                id="toggle-europe-button",
                n_clicks=0,
                style={"marginLeft": "20px"}
            )
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),
        html.Div(id="current-year-display", style={"fontSize": 24, "marginBottom": "20px"}),
        dcc.Graph(id="choropleth-map", figure=fig),
        dcc.Interval(
            id="interval-component",
            interval=500,  # Update every 0.5 second
            n_intervals=0  # Interval counter
        ),
        html.Div([
        html.H2("Welcome to the Mental Health Data Visualization!"),
        html.P("On this page, you'll find an interactive visualization that shows the evolution of mental health disorders across different countries, measured per thousand inhabitants. The choropleth map allows you to explore data for various metrics such as schizophrenia, anxiety, and depression, from 1990 until 2019."),
        html.P("To interact with the map, use the dropdown menu to select a metric. This will change the displayed metric on the map, letting you observe how different disorders have evolved over time across countries. As the data progresses through the years, the map will automatically update to reflect the changes."),
        html.P("Additionally, there's a button that lets you zoom in on Europe for a closer look at the regional data. You can toggle between a global view and a zoomed-in view of Europe by clicking the 'Zoom on Europe' button."),
        html.P("Feel free to explore the map and try to identify interesting patterns. For example, notice how mental health disorders in Greece saw a sharp rise during the country's massive economic downturn. Such patterns can help uncover the impact of major societal events on mental health."),
        html.P("We encourage you to play around with the different metrics and time periods to explore the trends and gain insights into how mental health challenges have evolved globally."),
    ], style={
        "padding": "20px",
        "backgroundColor": "#f9f9f9",
        "borderRadius": "8px",
        "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
        "fontFamily": "Arial, sans-serif",
        "color": "#333"
    })
    ])

@app.callback(
    [Output("choropleth-map", "figure"), Output("current-year-display", "children")],
    [Input("interval-component", "n_intervals"),
     Input("metric-dropdown", "value"),
     Input("toggle-europe-button", "n_clicks")]
)
def update_map_and_year(n_intervals, selected_metric, toggle_europe):
    target_column = normalized_columns[selected_metric]

    # Determine the year based on n_intervals
    year_index = n_intervals % len(unique_years)
    current_year = unique_years[year_index]

    # Filter data for the current year
    filtered_data = data[data["Year"] == current_year]

    # Create the map
    fig = px.choropleth(
        filtered_data,
        locations="Code",  # ISO-3 codes
        locationmode="ISO-3",  # Specify ISO-3 format
        color=target_column,
        color_continuous_scale="Viridis",
        range_color=[
            data[target_column].min(),
            data[target_column].max()
        ],
        title=f"{selected_metric} in {current_year}"
    )
    fig.update_layout(
        geo=dict(
            scope='europe' if toggle_europe % 2 == 1 else 'world',  # Toggle between Europe and World
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="white",
            projection_type="natural earth"
        ),
        width=1200,
        height=800,
        coloraxis_colorbar=dict(
            title="Normalized wrt 1990",
            thickness=20
        )
    )

    year_display = f"Current Year: {current_year}"
    return fig, year_display

# Page 1: Graph with Country and Metric Selection
def render_page_1():
    # Get list of countries (country codes)

    return html.Div([
        html.H1("Page 1: Detailed Analysis by Country and Metric"),
        html.Div([
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id="metric-dropdown-page-1",
                options=[{"label": metric, "value": metric} for metric in metrics],
                value=metrics[0],  # Default selection
                style={"width": "70%"}
            ),
            html.Label("Select Country:"),
            dcc.Dropdown(
                id="country-dropdown",
                options=[{"label": country, "value": country} for country in list_of_countries],
                value="USA",  # Default country
                style={"width": "70%", "marginTop": "20px"}
            ),
            
        ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "marginBottom": "20px"}),
        dcc.Graph(id="gdp-health-graph") ,
        
        
        html.Div([
            html.H2("Explore the Relationship Between GDP and Mental Health Diagnoses"),
            html.P("On this page, you'll find a graph comparing the GDP change and the prevalence of mental health diagnoses for a selected country, both normalized relative to 1990. This allows you to examine the potential correlation between a country's economic performance and the evolution of mental health disorders over time."),
            html.P("The graph presents two key metrics:"),
            html.Ul([
                html.Li("Normalized GDP change relative to 1990: This shows how the country's economy has grown or contracted since 1990."),
                html.Li("Normalized Mental Health Diagnoses: This shows how the prevalence of various mental health conditions has changed over time, relative to 1990 levels.")
            ]),
            html.P("You can use the dropdown menus to select a metric and a country. This will update the graph to show the corresponding time series data for the chosen country. By analyzing the trends, you can explore whether there are any patterns or correlations between GDP changes and mental health diagnoses."),
            html.P("Keep in mind that the data has been normalized, so you can directly compare the trends across different countries. Try to uncover any significant patterns—does a country's economic performance influence the rates of mental health diagnoses? For instance, economic recessions might be associated with increased mental health issues."),
            html.P("Feel free to experiment with different countries and metrics to deepen your understanding of the relationship between economic changes and mental health."),
        ], style={
            "padding": "20px",
            "backgroundColor": "#f9f9f9",
            "borderRadius": "8px",
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
            "fontFamily": "Arial, sans-serif",
            "color": "#333"
        })
        ])

@app.callback(
    Output("gdp-health-graph", "figure"),
    [Input("metric-dropdown-page-1", "value"),
     Input("country-dropdown", "value")]
)
def update_graph(selected_metric, selected_country):
    # Normalize the selected metric
    metric = "Normalized " + selected_metric

    # Filter the data based on the selected country and metric
    country_health = mental_health_df_filtered_by_country[mental_health_df_filtered_by_country['Code'] == selected_country].drop(columns=['Code', 'Entity'])
    country_health = np.transpose(country_health[['Year', metric]].values)

    country_gdp = gdp_df_filtered
    country_gdp = country_gdp[country_gdp['Country Code'] == selected_country].drop(columns='Country Code')

    # Create Plotly figure with dual axes (GDP and Mental Health Metric)
    fig = go.Figure()

    # Plotting Normalized GDP on the first axis (y-axis)
    fig.add_trace(go.Scatter(
        x=np.int32(country_gdp.columns),
        y=country_gdp.iloc[0],
        mode='lines+markers',
        name=f'{selected_country} - GDP',
        line=dict(color='blue'),
        yaxis='y1'
    ))

    # Plotting the Normalized Mental Health Metric (Schizophrenia, for example) on the second axis (y-axis)
    fig.add_trace(go.Scatter(
        x=country_health[0],
        y=country_health[1],
        mode='lines+markers',
        name=f'{selected_country} - {metric}',
        line=dict(color='red'),
        yaxis='y2'
    ))

    # Updating layout to create dual axes (one for GDP, one for Mental Health Metric)
    fig.update_layout(
        title=f'Time Series: Normalized GDP and {metric} for {selected_country}',
        xaxis=dict(title='Year'),
        yaxis=dict(
            title='Normalized GDP',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title=metric,
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0, y=1.0),
        template='plotly_white'  # A clean white background for the plot
    )

    return fig


# Page 2: Psychiatrists Data Plot
final_data = pd.read_csv('./eurostat/num_psych.csv')
final_data = final_data[final_data['unit'] == 'Per hundred thousand inhabitants']
final_data = final_data[final_data['TIME_PERIOD'] >= 2010]
pysch_data = final_data[final_data['med_spec'] == 'Psychiatrists']
selection=pysch_data['geo'].unique()

def render_page_2():
    
    return html.Div([
        html.H1("Page 2: Psychiatrists Data by Country"),
        html.Div([
            html.Label("Select Countries:"),
            dcc.Dropdown(
                id="country-dropdown-page-2",
                options=[{"label": country, "value": country} for country in selection],
                value=["Lithuania", "Spain", "France"],  # Default selection
                multi=True,  # Enable multiple selection
                style={"width": "70%", "marginTop": "20px"}
            ),
        ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "marginBottom": "20px"}),
        dcc.Graph(id="psychiatrist-graph"),
        html.Div([
            html.H2("Explore the Global Trend of Psychiatrists per Country"),
            html.P("This page provides a dynamic visualization of how countries have been increasing their number of psychiatrists over the past 10-15 years. It reflects a growing global awareness of mental health issues, as the demand for mental health professionals rises in most countries. This trend is a positive indicator of increasing focus on mental health care worldwide."),
            html.P("You can select multiple countries from the dropdown menu to compare their progress in terms of the number of psychiatrists per 100,000 inhabitants. The data is normalized to the year 2010, which makes it easier to compare trends across different countries, regardless of their baseline number of psychiatrists."),
            html.P("As you select more countries, you will see how their numbers have evolved, and whether this increase in psychiatrists correlates with any national events or trends, such as public health initiatives or economic shifts."),
            html.P("Feel free to explore different country selections, and see how global mental health care is evolving. Try to discover any patterns or interesting insights related to how different nations have responded to mental health challenges."),
            html.P("This page allows you to visualize the broader picture of mental health awareness worldwide. It might also highlight disparities between countries and help you understand how resources are being allocated to improve mental health care across regions."),
        ], style={
            "padding": "20px",
            "backgroundColor": "#f9f9f9",
            "borderRadius": "8px",
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
            "fontFamily": "Arial, sans-serif",
            "color": "#333"
        })


        
    ])

@app.callback(
    Output("psychiatrist-graph", "figure"),
    [Input("country-dropdown-page-2", "value")]
)
def update_psych_data(selected_countries):
    # Load the data
    final_data = pd.read_csv('./eurostat/num_psych.csv')
    final_data = final_data[final_data['unit'] == 'Per hundred thousand inhabitants']
    final_data = final_data[final_data['TIME_PERIOD'] >= 2010]
    pysch_data = final_data[final_data['med_spec'] == 'Psychiatrists']
    normalizing_data = pysch_data[pysch_data['TIME_PERIOD'] == 2010]
    normalizing_data.set_index(['geo'], inplace=True)
    pysch_data.set_index(['geo', 'TIME_PERIOD'], inplace=True)

    # Normalization step
    pysch_data['Normalized_OBS_VALUE'] = pysch_data['OBS_VALUE'] / normalizing_data['OBS_VALUE']
    pysch_data.reset_index(inplace=True)

    # Create the Plotly graph
    fig = go.Figure()

    for country in selected_countries:
        filtered_data = pysch_data[pysch_data['geo'] == country]

        # Add trace for each selected country
        fig.add_trace(go.Scatter(
            x=filtered_data['TIME_PERIOD'],
            y=filtered_data['Normalized_OBS_VALUE'],
            mode='lines+markers',
            name=country
        ))

    # Update layout to add titles, axis labels, etc.
    fig.update_layout(
        title="Normalized Psychiatrists per Country (2010 and onwards)",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Normalized Psychiatrists per 100,000 inhabitants"),
        template='plotly_white',
        legend=dict(title="Countries", x=0.1, y=0.9)
    )

    return fig

# Page rendering callback
@app.callback(
    Output("page-content", "children"),
    [Input("current-page", "data")]
)
def render_page(page_index):
    if page_index == 0:
        return render_page_0(metrics[0], False)
    elif page_index == 1:
        return render_page_1()
    elif page_index == 2:  # New page index for Psychiatrists data
        return render_page_2()

# Handle navigation between pages
@app.callback(
    Output("current-page", "data"),
    [Input("next-page-button", "n_clicks"), Input("previous-page-button", "n_clicks")],
    [State("current-page", "data")]
)
def update_page(next_clicks, previous_clicks, current_page):
    new_page = current_page + (next_clicks - previous_clicks)
    return max(0, min(new_page, 2))  # Ensure page stays in bounds


if __name__ == "__main__":
    app.run_server(debug=True)
