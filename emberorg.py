import streamlit as st
from streamlit_folium import folium_static
import folium
import json
import logging
from risk_assessment import RiskAssessmentAgent
from route_optimization import RouteOptimizationAgent
from instruction_generation import InstructionGenerationAgent
from scoring import score_instructions
from visualization import visualize_instructions_on_map

def load_map_data():
    evacuation_zones = []
    fire_perimeters = []
    try:
        # Load evacuation zone data
        with open('data/evacuation_zones.geojson', 'r') as file:
            evacuation_data = json.load(file)
        evacuation_zones = evacuation_data['features']

        # Log the number of evacuation zones loaded
        logging.info(f"Loaded {len(evacuation_zones)} evacuation zone features.")

        # Load fire perimeter data
        with open('data/fire_perimeters.geojson', 'r') as file:
            fire_data = json.load(file)
        fire_perimeters = fire_data['features']

        # Log the number of fire perimeters loaded
        logging.info(f"Loaded {len(fire_perimeters)} fire perimeter features.")

        logging.info("Successfully loaded map data.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON data: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in load_map_data: {e}")

    return evacuation_zones, fire_perimeters

def generate_evacuation_scenarios_with_user_input(evacuation_zones, fire_perimeters, num_scenarios, user_input):
    scenarios = []

    for _ in range(num_scenarios):
        # Use user input to select the evacuation zone and fire perimeter
        evacuation_zone = next((zone for zone in evacuation_zones if zone.get('properties', {}).get('name') == user_input.get('location')), None)
        fire_perimeter = next((perimeter for perimeter in fire_perimeters if perimeter.get('properties', {}).get('FIRE_NAME') == user_input.get('fire_name') and perimeter.get('properties', {}).get('YEAR_') == user_input.get('fire_year')), None)

        if evacuation_zone and fire_perimeter:
            # Generate a scenario description based on user input
            scenario_description = f"Evacuation scenario for {evacuation_zone['properties']['name']}:\n"
            scenario_description += f"Fire Name: {fire_perimeter['properties']['FIRE_NAME']}\n"
            scenario_description += f"Fire Year: {fire_perimeter['properties']['YEAR_']}\n"
            scenario_description += f"Evacuation Zone Type: {evacuation_zone['properties'].get('type', 'Unknown')}\n"

            # Calculate distance between evacuation zone and fire perimeter
            distance = evacuation_zone['geometry'].distance(fire_perimeter['geometry'])
            scenario_description += f"Distance to Fire: {distance:.2f} units\n"

            # Generate additional scenario details based on user input
            scenario_description += "Scenario Details:\n"
            scenario_description += f"- Number of People: {user_input.get('num_people', 'Unknown')}\n"
            # ... (additional scenario details based on user input)

            # Append the scenario to the list
            scenarios.append({
                'description': scenario_description,
                'location': evacuation_zone['properties']['name'],
                'fire_name': fire_perimeter['properties']['FIRE_NAME'],
                'fire_year': fire_perimeter['properties']['YEAR_']
            })
        else:
            logging.warning("No matching evacuation zone or fire perimeter found for the given user input.")

    return scenarios

def get_fires_with_evacuation_scenarios(evacuation_zones, fire_perimeters):
    fire_names = set()
    fire_years = set()

    for zone in evacuation_zones:
        location = zone.get('properties', {}).get('name')
        if location:
            matching_fires = [perimeter for perimeter in fire_perimeters if perimeter.get('properties', {}).get('FIRE_NAME') and perimeter.get('properties', {}).get('YEAR_')]
            for fire in matching_fires:
                fire_names.add(fire['properties']['FIRE_NAME'])
                fire_years.add(fire['properties']['YEAR_'])

    return list(fire_names), list(fire_years)

def main():
    st.set_page_config(page_title="EMBER - Wildfire Evacuation Recommendation System", layout="wide")
    st.title("EMBER - Wildfire Evacuation Recommendation System")

    # User input section
    st.sidebar.header("Scenario Input")
    location = st.sidebar.text_input("Location")
    fire_name = st.sidebar.text_input("Fire Name")
    fire_year = st.sidebar.number_input("Fire Year", min_value=2000, max_value=2023, step=1)
    num_people = st.sidebar.number_input("Number of People", min_value=1, max_value=10000, step=1)
    # ... (additional user input fields)

    # Refresh button
    if st.button("Refresh"):
        st.experimental_rerun()

    # Load map data
    evacuation_zones, fire_perimeters = load_map_data()

    # Get fires with evacuation scenarios
    fire_names, fire_years = get_fires_with_evacuation_scenarios(evacuation_zones, fire_perimeters)

    # Display fires with evacuation scenarios
    st.sidebar.header("Fires with Evacuation Scenarios")
    st.sidebar.write("Fire Names:")
    for name in fire_names:
        st.sidebar.write(f"- {name}")
    st.sidebar.write("Fire Years:")
    for year in fire_years:
        st.sidebar.write(f"- {year}")

    # Generate evacuation scenarios based on user input
    num_scenarios = 1
    user_input = {
        'location': location,
        'fire_name': fire_name,
        'fire_year': fire_year,
        'num_people': num_people,
        # ... (additional user input fields)
    }
    evacuation_scenarios = generate_evacuation_scenarios_with_user_input(evacuation_zones, fire_perimeters, num_scenarios, user_input)

    if not evacuation_scenarios:
        st.warning("No evacuation scenarios found for the given user input. Please try again using the available fire names and years listed in the sidebar.")
        return

    # Initialize agents
    risk_assessment_agent = RiskAssessmentAgent()
    route_optimization_agent = RouteOptimizationAgent()
    instruction_generation_agent = InstructionGenerationAgent()

    # Define personas
    personas = [
        {'name': 'Persona 1', 'concern': 'safety', 'trait': 'cautious'},
        {'name': 'Persona 2', 'concern': 'family', 'trait': 'caring'},
        {'name': 'Persona 3', 'concern': 'pets', 'trait': 'responsible'}
    ]

    # Assess risk levels
    risk_levels, prioritized_zones = risk_assessment_agent.assess_risk(evacuation_zones)

    # Optimize evacuation routes
    optimized_routes = route_optimization_agent.optimize_routes(evacuation_zones, risk_levels, prioritized_zones)

    # Generate and score instructions for each scenario
    instructions_list = []
    for scenario in evacuation_scenarios:
        instructions = instruction_generation_agent.generate_instructions(optimized_routes, user_input)
        instructions_with_scores = score_instructions(instructions, scenario, personas)
        instructions_list.append(instructions_with_scores)

    try:
        # Visualize instructions on the map
        map = visualize_instructions_on_map(evacuation_zones, fire_perimeters, evacuation_scenarios, instructions_list, risk_levels, optimized_routes)
        folium_static(map)
    except Exception as e:
        st.error(f"An error occurred while visualizing the instructions: {str(e)}")

    # Display the generated instructions
    for i, (scenario, instructions) in enumerate(zip(evacuation_scenarios, instructions_list)):
        st.subheader(f"Scenario {i+1}")
        st.write(scenario['description'])
        st.write("Generated Instructions:")
        for j, instruction in enumerate(instructions):
            st.write(f"{j+1}. {instruction['instruction']}")
            st.write(f"   Total Score: {instruction['total_score']:.2f}")
            st.write(f"   Safety Score: {instruction['safety_score']:.2f}")
            st.write(f"   Specificity Score: {instruction['specificity_score']:.2f}")
            st.write(f"   Clarity Score: {instruction['clarity_score']:.2f}")
            st.write(f"   Feedback Score: {instruction['feedback_score']:.2f}")
        st.write("---")

    # Feedback mechanism
    feedback = st.text_input("Provide feedback on the generated instructions")
    rating = st.slider("Rate the instructions", min_value=1, max_value=5, value=3, step=1)
    if st.button("Submit Feedback"):
        instruction_generation_agent.process_feedback((feedback, rating))

if __name__ == "__main__":
    main()main__":
    main()