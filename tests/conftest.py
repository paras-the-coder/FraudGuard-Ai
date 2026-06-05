import os
import sys
import pytest

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def sample_claim():
    """
    Provides a dictionary containing the columns and values representing a standard
    legitimate claim (from the actual dataset). This serves as the base input for testing
    the data preprocessing, feature engineering, and model inference pipeline.
    """
    return {
        "months_as_customer": 134,
        "age": 29,
        "policy_number": 687698,
        "policy_bind_date": "9/6/2000",
        "policy_state": "OH",
        "policy_csl": "100/300",
        "policy_deductable": 2000,
        "policy_annual_premium": 1413.14,
        "umbrella_limit": 5000000,
        "insured_zip": 430632,
        "insured_sex": "FEMALE",
        "insured_education_level": "PhD",
        "insured_occupation": "sales",
        "insured_hobbies": "board-games",
        "insured_relationship": "own-child",
        "capital-gains": 35100,
        "capital-loss": 0,
        "incident_date": "2/22/2015",
        "incident_type": "Multi-vehicle Collision",
        "collision_type": "Rear Collision",
        "incident_severity": "Minor Damage",
        "authorities_contacted": "Police",
        "incident_state": "NY",
        "incident_city": "Columbus",
        "incident_location": "7121 Francis Lane",
        "incident_hour_of_the_day": 7,
        "number_of_vehicles_involved": 3,
        "property_damage": "NO",
        "bodily_injuries": 2,
        "witnesses": 3,
        "police_report_available": "NO",
        "total_claim_amount": 34650,
        "injury_claim": 7700,
        "property_claim": 3850,
        "vehicle_claim": 23100,
        "auto_make": "Dodge",
        "auto_model": "RAM",
        "auto_year": 2007
    }
