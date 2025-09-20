from playwright.sync_api import sync_playwright
import pandas as pd
import re, time

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

    # Add pollutant data to the new dataframe
    for n, i in enumerate(range(1, 12, 2)):
        pollutant = page.locator('.pollutant-concentration').nth(i)
        pollutant.wait_for(state='attached')

        data[names[n]] = float(pollutant.inner_text().split()[0])

    # Go to weather forecasts page to access weather data
    page.goto(
        'https://www.accuweather.com/en/pk/karachi/261158/current-weather/261158'
    )

    cards = page.locator('.detail-item.spaced-content').all_inner_texts()
    weather = ['wind_speed_km', 'humidity_percent', 'pressure_mb']

    # Add temperature data to the new dataframe
    TEMP = page.locator('.display-temp').first
    TEMP.wait_for(state='attached')
    UNIT = TEMP.inner_text().split('°')[1]
    MAGNITUDE = float(TEMP.inner_text().split('°')[0])
    if UNIT == 'F':
        data['temp'] = round((MAGNITUDE-32) * 5/9 , 1)
    else:
        data['temp'] = round(MAGNITUDE, 1)
    
    # Getting only the wind speed, humidity percentage and pressure (in mb)
    cards = [ cards[n] for n, i in enumerate(cards) if 'Wind Gusts' in i or 'Indoor Humidity' in i or 'Pressure' in i ]

    for n, i in enumerate(cards):

        # Convert from in to mb (international standard) for pressure and mph to km/h
        # Converting imperial units to metric units
        if weather[n] == 'pressure_mb':
            MAGNITUDE = float(i.split()[2::][0])
            UNIT = i.split()[2::][1]
            if UNIT == 'in':
                data[weather[n]] = round(MAGNITUDE * 33.8639, 1)
            else:
                data[weather[n]] = round(MAGNITUDE, 1)
        elif weather[n] == 'wind_speed_km':
            MAGNITUDE = float(i.split()[2])
            UNIT = i.split()[3]
            if UNIT == 'km/h':
                data[weather[n]] = round(MAGNITUDE, 1)
            else:
                data[weather[n]] = round(MAGNITUDE * 1.609, 1)
        else:
            data[weather[n]] = round(float(re.findall(r'\d+', i)[0]), 1)

    # Convert python dictionary to Pandas DataFrame
    row = pd.DataFrame([data])

    # Append data to CSV file
    with open('aqi-data.csv', 'a') as f:
        row = ','.join(str(value) for value in data.values())
        f.write('\n'+row)

    browser.close()