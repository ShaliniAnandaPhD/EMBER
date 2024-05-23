# EMBER - Wildfire Evacuation Recommendation System 🔥🆘

EMBER is an innovative and potentially life-saving AI-powered system that utilizes real-time data, advanced risk assessment techniques, optimized route planning, and cutting-edge natural language generation to deliver personalized evacuation instructions during the critical moments of a wildfire emergency. 🌳🔥🏃‍♂️

## Table of Contents
- [Key Features](#key-features) ✨
- [File Structure](#file-structure) 📂
  - [emberorg.py](#emberorgpy)
  - [risk_assessment.py](#risk_assessmentpy)
  - [route_optimization.py](#route_optimizationpy)
  - [instruction_generation.py](#instruction_generationpy)
  - [scoring.py](#scoringpy)
  - [visualization.py](#visualizationpy)
  - [RAG Approaches](#rag-approaches)
    - [Wildfire_RAG.py](#wildfire_ragpy)
    - [WildfireEvacPipeline.py](#wildfireevacpipelinepy)
    - [WildFireEvacRAG.py](#wildfireevacragpy)
  - [Supporting Files](#supporting-files)
    - [RAG.py](#ragpy)
    - [LlamaIndexQA.py](#llamaindexqapy)
    - [rl_framework.py](#rl_frameworkpy)
- [Installation](#installation) 💻
- [Usage](#usage) 🚀
- [Contributing](#contributing) 🤝
- [License](#license) 📜


## Key Features ✨
- 🌡️ Intelligent risk assessment and prioritization of evacuation zones based on real-time data and map information
- 🗺️ Optimized route planning considering traffic conditions, potential hazards, and risk levels
- 🗣️ Personalized evacuation instruction generation using state-of-the-art natural language models
- 🏆 Scoring and ranking of generated instructions based on safety, specificity, clarity, and user feedback
- 🌐 Interactive visualization of evacuation zones, fire perimeters, and recommended routes on a user-friendly map
- 🧠 Integration of reinforcement learning to continuously improve instruction generation based on valuable user feedback
- 📈 Comprehensive monitoring and evaluation of system performance and generated instructions

Data and Data Formats
Data Requirements

EMBER requires various types of data to function effectively:

- Evacuation Zones: GIS data outlining evacuation zones.
- Fire Perimeters: Real-time data on current wildfire perimeters.
- Traffic Conditions: Live traffic data to optimize evacuation routes.
- User Inputs: Information such as location, fire name, fire year, and number of people.

Data Formats
- Evacuation Zones: Shapefile (.shp), GeoJSON (.geojson)
- Fire Perimeters: GeoJSON (.geojson), KML (.kml)
- Traffic Conditions: JSON format with live traffic updates
- User Inputs: JSON or CSV format
  
Setting Up Data

- Evacuation Zones: Obtain GIS data files in Shapefile or GeoJSON format that outline the evacuation zones.
- Fire Perimeters: Acquire real-time wildfire perimeter data in GeoJSON or KML format from reliable sources such as government agencies or fire monitoring services.
- Traffic Conditions: Integrate live traffic data in JSON format from APIs provided by traffic monitoring services.
- User Inputs: Collect user input data in JSON or CSV format, ensuring it includes all necessary information for generating personalized evacuation instructions.


## File Structure 📂

### emberorg.py
The heart of the EMBER system, this Streamlit application brings together all the components to create a seamless user experience. It efficiently loads map data, generates evacuation scenarios based on user input, coordinates the risk assessment, route optimization, and instruction generation agents, and visualizes the instructions on an interactive map. The application also includes a feedback mechanism, allowing users to rate the generated instructions and contribute to the system's continuous improvement. 💡

### risk_assessment.py
The `RiskAssessmentAgent` class, defined in this file, plays a crucial role in assessing the risk levels of evacuation zones based on real-time data and map information. Leveraging a sophisticated pre-trained risk assessment model, it accurately predicts risk levels and prioritizes evacuation zones accordingly, ensuring that the most vulnerable areas receive the attention they need. 🚨

### route_optimization.py
The `RouteOptimizationAgent` class, found in this file, is responsible for finding the safest and most efficient evacuation routes. By considering critical factors such as traffic conditions, potential hazards, and risk levels, it generates optimized routes for each evacuation zone, minimizing the risk and ensuring a smooth evacuation process. 🚗

### instruction_generation.py
The `InstructionGenerationAgent` class, defined in this file, lies at the core of EMBER's ability to provide personalized evacuation instructions. Utilizing advanced natural language models, it takes optimized routes and user inputs to generate clear, concise, and actionable instructions. With the integration of reinforcement learning, the agent continuously improves its generated instructions based on valuable user feedback, adapting to the unique needs of each individual. 🌍

### scoring.py
This file contains essential functions for scoring and ranking the generated instructions based on critical criteria such as safety, specificity, clarity, and user feedback. By employing a weighted scoring system, it calculates a comprehensive score for each instruction, ensuring that the most relevant and effective instructions are prioritized and presented to the users. 🎯

### visualization.py
The `visualize_instructions_on_map` function, housed in this file, brings the evacuation scenarios to life by creating an intuitive and interactive visualization. Utilizing the powerful Folium library, it seamlessly integrates evacuation zones, fire perimeters, recommended routes, and generated instructions onto a user-friendly map. With markers, polygons, and informative popups, users can easily grasp the evacuation situation and make informed decisions. 🗺️

## RAG Approaches

### Wildfire_RAG.py
The `WildfireRAG` class represents a cutting-edge Retrieval-Augmented Generation (RAG) system specifically designed for generating wildfire evacuation instructions. By seamlessly integrating document retrieval from an Elasticsearch index, instruction generation using a pre-trained GPT-2 model, and personalized query processing, this approach delivers highly relevant and tailored instructions. The integration with FastAPI allows for easy deployment and accessibility through a web API endpoint. 🌐🔥

### WildfireEvacPipeline.py
The `WildfireEvacPipeline` class offers a streamlined pipeline approach for generating wildfire evacuation instructions. It combines the power of a pre-trained sentence-transformer model for efficient document retrieval and a pre-trained seq2seq model for generating high-quality instructions. By retrieving relevant documents from an Elasticsearch index and leveraging the user's query, this pipeline produces instructions that are both informative and specific to the user's needs. 🚰🔥

### WildFireEvacRAG.py
The `WildfireEvacRAG` class presents a innovative two-stage Retrieval-Augmented Generation (RAG) system for delivering wildfire evacuation instructions. In the retrieval stage, it employs BM25 scoring to identify the most relevant documents from a pre-indexed corpus. These retrieved documents, along with the user's query, are then fed into a fine-tuned BART model in the generation stage, producing highly accurate and context-aware evacuation instructions. Implemented as a command-line application, this approach offers flexibility and ease of use. 🌿🔥

## Supporting Files

### RAG.py
The `WildfireEvacuationRAG` class in this file showcases a powerful RAG system that seamlessly integrates document retrieval, instruction generation, and reinforcement learning. By leveraging a sentence transformer model for relevant document retrieval, a GPT-2 model for generating instruction candidates, and a BERT-based scoring model for ranking instructions, this system produces high-quality evacuation instructions. The incorporation of user feedback through reinforcement learning ensures that the system continuously adapts and improves its output. 🧠💡

### LlamaIndexQA.py
This file demonstrates the impressive capabilities of the LlamaIndex library in generating evacuation scenarios and instructions. By creating a document index using the GPTSimpleVectorIndex and querying the index with a carefully crafted prompt, it obtains highly relevant responses. The prompt incorporates detailed information about forest characteristics, campers, exits, evacuation centers, and a specific wildfire scenario, enabling the generation of comprehensive and tailored evacuation instructions. 🦙💡

### rl_framework.py
The `RLAgent` class, defined in this file, serves as a reinforcement learning agent that plays a vital role in updating the instruction generation model based on user feedback. By employing a pre-trained sentiment analysis model, it accurately calculates rewards based on user feedback, allowing the system to adapt and improve its generated instructions over time. Through the policy gradient method, the agent effectively updates the model parameters, ensuring that the system continually learns and refines its output. 🧠📈

## Installation 💻
1. Clone the repository:
   ```
   git clone https://github.com/your-username/ember.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary data files and pre-trained models (evacuation zones, fire perimeters, risk assessment model, route optimization model, language models).

4. Run the Streamlit application:
   ```
   streamlit run emberorg.py
   ```

## Usage 🚀
1. Access the EMBER application through the provided Streamlit interface.
2. Input the required information, such as location, fire name, fire year, and number of people.
3. The system will generate personalized evacuation instructions based on the provided inputs and real-time data.
4. Interact with the generated instructions, provide feedback, and explore the visualized evacuation routes on the map.
5. The system will continuously learn and improve based on user feedback and interactions.

## Contributing 🤝
Contributions to EMBER are always welcome! If you encounter any issues, have ideas for improvements, or want to add new features, please don't hesitate to open an issue or submit a pull request on the GitHub repository. Together, we can make EMBER even more powerful and effective in helping people stay safe during wildfire emergencies. 🙌

## License 📜
This project is licensed under the [MIT License](LICENSE), which allows for open collaboration and encourages the community to build upon and enhance the EMBER system.
