Note: Synatx for few are referred from offical documentation https://openrouter.ai/docs/api/reference/overview
## Environment Variables

Create a `.env` file in the project root containing:


OPENROUTER_API_KEY=your_api_key_here


The notebook loads the API key using the `python-dotenv` package.
Temperature was set to 0 because structured JSON extraction requires consistent and predictable outputs. At temperature 0, the model selects the highest-probability next token, reducing randomness and improving reproducibility. A temperature of 0.7 introduces more variability because the model samples from a broader probability distribution. In this experiment, both temperatures produced similar outputs because the prompt contained strict JSON formatting instructions and few-shot examples that constrained the model response.


Task 3- Structured output :

[{'Input': {'City': 'Lucknow',
   'Date': '2018-05-23',
   'season': 'Summer',
   'AQI_Bucket': 'Poor',
   'avg_temp': 36.95,
   'min_temp': 29.15,
   'max_temp': 44.05,
   'wind_speed': 11.1,
   'air_pressure': 1009.9,
   'AQI': 257.0,
   'ONI': -0.13},
  'Output': {'city': 'Lucknow',
   'aqi': 257.0,
   'category': 'Poor',
   'season': 'Summer',
   'temperature': 36.95,
   'summary': 'High AQI may affect health and outdoor exposure should be reduced.'}},
 {'Input': {'City': 'Chennai',
   'Date': '2019-11-23',
   'season': 'Post-monsoon',
   'AQI_Bucket': 'Satisfactory',
   'avg_temp': 25.7,
   'min_temp': 23.2,
   'max_temp': 30.3,
   'wind_speed': 5.4,
   'air_pressure': 1013.6,
   'AQI': 64.0,
   'ONI': 0.58},
  'Output': {'city': 'Chennai',
   'aqi': 64.0,
   'category': 'Satisfactory',
   'season': 'Post-monsoon',
   'temperature': 25.7,
   'summary': 'Air Quality is Satisfactory.'}},
 {'Input': {'City': 'Chennai',
   'Date': '2018-08-27',
   'season': 'Monsoon',
   'AQI_Bucket': 'Satisfactory',
   'avg_temp': 29.5,
   'min_temp': 26.0,
   'max_temp': 35.7,
   'wind_speed': 11.2,
   'air_pressure': 1005.1,
   'AQI': 70.0,
   'ONI': 0.27},
  'Output': {'city': 'Chennai',
   'aqi': 70.0,
   'category': 'Satisfactory',
   'season': 'Monsoon',
   'temperature': 29.5,
   'summary': 'Air Quality is Satisfactory.'}}]

   All three test inputs produced valid JSON outputs and successfully passed schema validation. No JSON parsing or schema validation failures were observed. This indicates that the few-shot examples and strict JSON output instructions helped the LLM consistently follow the expected structure.

pii_input = """
Extract AQI information.
Contact person: divyaa@gmail.com
"""

Input blocked: PII detected.
None



{
  "City": "Lucknow",
  "Date": "2018-05-23",
  "season": "Summer",
  "AQI_Bucket": "Poor",
  "avg_temp": 36.95,
  "min_temp": 29.15,
  "max_temp": 44.05,
  "wind_speed": 11.1,
  "air_pressure": 1009.9,
  "AQI": 257.0,
  "ONI": -0.13,
  "summary": "High AQI may affect health and outdoor exposure should be reduced."
}


The end-to-end pipeline was successfully executed on three AQI dataset records. All three inputs passed the PII guardrail because no personally identifiable information was detected. The LLM generated structured JSON outputs for all records, and each response successfully passed JSON parsing and schema validation. This demonstrates a complete workflow combining prompt engineering, structured output handling, and safety validation.
