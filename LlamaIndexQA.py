# Installation and Imports
pip install llama-index
import numpy as np
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader, ServiceContext

# Forest Characteristics
grid_size = 50  # Increased grid size for a larger forest area
density = np.random.rand(grid_size, grid_size)  # Random density in each cell (trees, undergrowth)
elevation = np.random.randint(100, 1000, size=(grid_size, grid_size))  # Random elevation in each cell (in meters)

# Campers
num_campers = 20  # Increased number of campers
camper_locations = np.random.randint(1, grid_size - 1, size=(num_campers, 2))  # (x, y) coords, avoiding edges
camper_health = np.random.choice(["Good", "Fair", "Poor"], size=num_campers, p=[0.7, 0.2, 0.1])  # Camper health status

# Exits and Evacuation Centers
num_exits = 6
exits = np.random.randint(0, grid_size, size=(num_exits, 2))  # Random exit locations
evacuation_centers = np.random.randint(0, grid_size, size=(3, 2))  # Random evacuation center locations

# Scenario Generation
def generate_scenario():
    fire_origin = np.random.randint(1, grid_size - 1, size=2)  # Starting point of the fire, avoiding edges
    wind_speed = np.random.randint(10, 51)  # Wind speed in km/h
    wind_direction = np.random.uniform(0, 2 * np.pi)  # Wind direction in radians
    fire_intensity = np.random.choice(["Low", "Moderate", "High", "Extreme"])  # Fire intensity levels
    smoke_level = np.random.choice(["Low", "Moderate", "High"])  # Smoke level
    return fire_origin, wind_speed, wind_direction, fire_intensity, smoke_level

scenario = generate_scenario()

# Prompt Generation
prompt = f"""
A {scenario[3]} intensity forest fire has started at coordinates ({scenario[0][0]}, {scenario[0][1]}) in a {grid_size}x{grid_size} area.
Wind is blowing at a speed of {scenario[1]} km/h in the direction of {np.degrees(scenario[2]):.2f} degrees (clockwise from North).
There are {num_campers} campers dispersed in the forest, with varying health conditions: {', '.join(camper_health)}.
The known safe exits are located at coordinates: {', '.join(str(exit) for exit in exits)}.
Evacuation centers have been set up at coordinates: {', '.join(str(center) for center in evacuation_centers)}.
The forest has varying tree density and elevation, which can impact fire spread and evacuation efforts.
The smoke level in the area is currently {scenario[4]}.

Describe potential evacuation routes and behaviors of the campers, considering their health conditions, the fire spread, wind conditions, elevation, and the location of safe exits and evacuation centers.
Discuss the challenges faced by the authorities in coordinating the evacuation efforts given the limited information about camper locations and the varying terrain.
Propose strategies for efficient communication and resource allocation during the evacuation process.
"""

# Create a document from the prompt
documents = [Document(prompt)]

# Index Creation and Query
# Create an index
service_context = ServiceContext.from_defaults()
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

# Query the index
response = index.query(prompt)
print(response)