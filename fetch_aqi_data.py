from playwright.sync_api import sync_playwright
import pandas as pd
from dotenv import load_dotenv
import re, time

load_dotenv()

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()
    page.goto(
        'https://www.accuweather.com/en/pk/karachi/261158/air-quality-index/261158',
    )

    # A template for a row in the dataset
    data = {
        'hour': None,
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

    names = ['pm_10', 'pm_2_5', 'no_2', 'o_3', 'so_2', 'co']

    # Get current hour in UTC
    utc_hour = time.gmtime().tm_hour
    # Convert to Pakistan Time (UTC+5)
    pkt_hour = (utc_hour + 5) % 24
    data['hour'] = pkt_hour

    # Add current AQI to the new dataframe
    AQI = page.locator('.aq-number').first
    AQI.wait_for(state='attached')
    data['aqi'] = float(AQI.inner_text())

    # Add current temperature to the new dataframe
    TEMP = page.locator('.header-temp').first
    TEMP.wait_for(state='attached')
    
    # Converting Fahrenheit to Celcius
    F = float(TEMP.inner_text().split('°')[0])
    data['temp'] = round( float((F - 32) * 5 / 9), 1)

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
    weather = ['wind_speed_km', 'humidity_percent', 'pressure_mb']
    
    # Getting only the wind speed, humidity % and pressure (in mb)
    cards = [ cards[2], cards[3], cards[6] ]
    for n, i in enumerate(cards):
        data[weather[n]] = float(re.findall(r'\d+', i)[0])

    # Convert python dictionary to Pandas DataFrame
    row = pd.DataFrame([data])

    with open('aqi-data.csv', 'a') as f:
        row = ','.join(str(value) for value in data.values())
        f.write('\n'+row)

    browser.close()