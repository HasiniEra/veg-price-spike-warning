# Vegetable price-spike early warning (Streamlit)

Chance that a vegetable's Dambulla wholesale price rises by more than 25% within 2 weeks,
from price history and weather. Model: Quadratic Discriminant Analysis + isotonic calibration
(retrained from the data on start-up, takes under a second).

## Run
    pip install -r requirements.txt
    streamlit run app.py

Data: Central Bank of Sri Lanka daily vegetable prices; NASA POWER weather for 7 Sri Lankan locations.
