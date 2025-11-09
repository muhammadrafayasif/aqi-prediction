from playwright.sync_api import sync_playwright
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import re, os, hopsworks

# Load .env variables to communicate with HopsWorks
load_dotenv()

# Login to HopsWorks and get our feature store
project = hopsworks.login(
    api_key_value = os.getenv('API_KEY')
)
fs = project.get_feature_store()

# Get or create a feature group containing our features for AQI prediction
fg = fs.get_or_create_feature_group(
    name="aqi_feature_pipeline",
    version=2,
    online_enabled=True,
    primary_key=["timestamp_str"],
    description="A feature pipeline for storing AQI, environmental pollutants and weather data to the feature store."
)

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)

    page = browser.new_page()
    
    # Convert units from imperial to metric system
    page.goto(
        'https://www.accuweather.com/en/settings',
    )
    page.select_option("select#unit", "C")


    page.goto(
        'https://www.accuweather.com/en/pk/karachi/261158/air-quality-index/261158',
    )

    # A template for a row in the dataset
    data = {
        'timestamp': None,
        'temp': None,
        'wind_speed': None,
        'wind_gusts': None,
        'humidity_percent': None,
        'dew_point': None,
        'pressure': None,
        'cloud_cover': None,
        'visibility': None,
        'pm_10': None,
        'pm_2_5': None,
        'no_2': None,
        'o_3' : None,
        'so_2': None,
        'co': None,
        'aqi': None
    }

    names = ['pm_10', 'pm_2_5', 'no_2', 'o_3', 'so_2', 'co']

    # Add current data and time to the new dataframe
    data['timestamp'] = datetime.now().replace(minute=0, second=0, microsecond=0)

    # Add current AQI to the new dataframe
    AQI = page.locator('.aq-number').first
    AQI.wait_for(state='attached')
    data['aqi'] = float(AQI.inner_text())

    # Add pollutant data to the new dataframe
    for n, i in enumerate(range(1, 12, 2)):
        pollutant = page.locator('.pollutant-concentration').nth(i)
        pollutant.wait_for(state='attached')

        data[names[n]] = float(pollutant.inner_text().split()[0])

    # Go to weather forecasts page to access weather data
    page.goto(
        'https://www.accuweather.com/en/pk/karachi/261158/current-weather/261158'
    )

    # Add temperature data to the new dataframe
    TEMP = page.locator('.display-temp').first
    TEMP.wait_for(state='attached')
    MAGNITUDE = float(TEMP.inner_text().split('°')[0])
    data['temp'] = round(MAGNITUDE, 1)
    
    # Getting the temperature data required
    cards = page.locator('.detail-item.spaced-content').all_inner_texts()
    
    # Only getting the required features we need, the following are not needed for our model
    exclude = ['UV Index', 'RealFeel', 'Indoor Humidity', 'Cloud Ceiling']
    cards = [ cards[n] for n, i in enumerate(cards) if not any(k in i for k in exclude)]

    # Getting weather data from the website and filtering any noise
    weather = [
        'wind_speed',
        'wind_gusts',
        'humidity_percent',
        'dew_point',
        'pressure',
        'cloud_cover',
        'visibility'
    ]

    for n, i in enumerate(cards):
        data[weather[n]] = round(float(re.findall(r'\d+', i)[0]), 1)

    # Add timestamp_str as primary key for online feature store
    data['timestamp_str'] = pd.to_datetime(data["timestamp"]).astype("int64") // 10**9

    # Convert python dictionary to Pandas DataFrame
    row = pd.DataFrame([data])

    # Append current data to our feature store
    fg.insert(row, wait=True)

    browser.close()