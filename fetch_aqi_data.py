from playwright.sync_api import sync_playwright
import pandas as pd
from dotenv import load_dotenv
import re, os, hopsworks

load_dotenv()

# Connect to Hopsworks data store
project = hopsworks.login(
    api_key_value=os.getenv("API_KEY"),
    project='aqiData',
    host='c.app.hopsworks.ai'
)

# Access your feature store
fs = project.get_feature_store(name='aqidata_featurestore')
fg = fs.get_feature_group(name="aqi_data", version=1)

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()
    page.goto(
        'https://www.accuweather.com/en/pk/karachi/261158/air-quality-index/261158',
    )

    # A template for a row in the dataset
    data = {
        'temp': None,
        'wind_speed_km': None,
        'humidity_percent': None,
        'pressure_mb': None,
        'pm_10': None,
        'pm_2_5': None,
        'no_2': None,
        'o_3' : None,
        'so_2': None,
        'co': None,
        'aqi': None
    }

    names = ['PM_10', 'PM_2.5', 'NO_2', 'O_3', 'SO_2', 'CO']

    # Add current AQI to the new dataframe
    AQI = page.locator('.aq-number').first
    AQI.wait_for(state='attached')
    data['AQI'] = float(AQI.inner_text())

    # Add current temperature to the new dataframe
    TEMP = page.locator('.header-temp').first
    TEMP.wait_for(state='attached')
    data['TEMP'] = float(TEMP.inner_text().replace('°C', ''))

    # Add pollutant data to the new dataframe
    for n, i in enumerate(range(1, 12, 2)):
        pollutant = page.locator('.pollutant-concentration').nth(i)
        pollutant.wait_for(state='attached')

        # According to AQI standards, CO must be in ppm
        # AccWeather provides CO in ug/(m^3), which is converted to ppm through a formula
        data[names[n]] = round(float(

            float(pollutant.inner_text().split()[0])
                            *
            (1 if names[n]!='CO' else 24.45 / (1000 * 28.01))

        ), 3)

    # Go to weather forecasts page to access weather data
    page.goto(
        'https://www.accuweather.com/en/pk/karachi/261158/current-weather/261158'
    )

    cards = page.locator('.detail-item.spaced-content').all_inner_texts()
    weather = ['WIND_SPEED_KM', 'HUMIDITY %', 'PRESSURE_MB']
    
    # Getting only the wind speed, humidity % and pressure (in mb)
    cards = [ cards[2], cards[3], cards[6] ]
    for n, i in enumerate(cards):
        data[weather[n]] = float(re.findall(r'\d+', i)[0])

    # Convert python dictionary to Pandas DataFrame
    row = pd.DataFrame([data])

    browser.close()

fg.insert(row)