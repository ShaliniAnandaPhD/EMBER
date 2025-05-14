# Attribution Notice

This repository documents original work by **Shalini Ananda** for a wildfire detection and resilience project titled **Ember**, developed prior to and during a hackathon hosted at **AGI House, Hillsborough** on **January 18, 2025**.

During the event, the project concept, name, and portions of the GitHub repository were accessed via a shared team submission spreadsheet and subsequently used by another participant or team without consent or attribution. Despite my system breaking during the demo, I was called to present first and delivered the core vision and intent behind Ember in front of the full group.

After the event, I publicly posted the full GitHub repository to the official hackathon Discord for transparency and record. While another team went on to win recognition using material closely resembling this project, **Ember was originated, named, and built by me.**

This repository serves as the canonical source of the original Ember project.

Please respect its creative and technical origins. If you reference or adapt this work, **you must include visible attribution to this repository and author.**

A modified MIT License applies (see `LICENSE` file).



# EMBER - Wildfire Evacuation Recommendation System 🔥🆘

EMBER is a potentially life-saving AI-powered system that utilizes real-time data, advanced risk assessment techniques, optimized route planning, and cutting-edge natural language generation to deliver personalized evacuation instructions during the critical moments of a wildfire emergency. 🌳🔥🏃‍♂️

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



---

# **File Structure**

### **emberorg.py**
This file serves as the central application for the EMBER system, developed using Streamlit to provide an interactive user interface. It integrates multiple system components, including risk assessment, route optimization, and instruction generation, to deliver personalized evacuation recommendations. The application dynamically processes user inputs, retrieves relevant geospatial and real-time hazard data, and visualizes evacuation scenarios on an interactive map. Additionally, it includes a feedback mechanism that enables users to rate the generated instructions, contributing to continuous system refinement.

### **risk_assessment.py**
This module implements the `RiskAssessmentAgent` class, which is responsible for evaluating the relative risk levels of various evacuation zones. The agent employs a pre-trained risk assessment model to analyze geospatial data and real-time wildfire conditions, prioritizing evacuation efforts based on hazard severity and population vulnerability. The goal is to ensure that high-risk areas receive expedited guidance and intervention.

### **route_optimization.py**
The `RouteOptimizationAgent` class, defined in this module, is designed to compute the most efficient and safest evacuation routes. The algorithm integrates real-time traffic conditions, geographical constraints, and fire progression models to dynamically adjust routes, minimizing exposure to hazards while optimizing travel time. This approach enhances evacuation efficiency and reduces congestion in high-risk areas.

### **instruction_generation.py**
The `InstructionGenerationAgent` class constitutes a core component of the system, facilitating the generation of personalized evacuation instructions. This module leverages state-of-the-art natural language processing (NLP) models to transform optimized route data into clear and context-aware evacuation directives. The system incorporates reinforcement learning mechanisms to iteratively refine instruction clarity, specificity, and effectiveness based on user feedback.

### **scoring.py**
This module provides a structured evaluation framework for ranking evacuation instructions. The scoring function applies a weighted system that considers safety, specificity, clarity, and user feedback. By systematically ranking instructions, this approach ensures that users receive the most relevant and actionable recommendations.

### **visualization.py**
The `visualize_instructions_on_map` function, implemented within this module, enables the graphical representation of evacuation plans. Using the Folium library, this function integrates geospatial data, fire perimeters, recommended routes, and generated instructions into an interactive map interface. This visualization aids decision-making by providing an intuitive and comprehensive overview of the evacuation scenario.

---

# **Retrieval-Augmented Generation (RAG) Approaches**

### **Wildfire_RAG.py**
This module defines the `WildfireRAG` class, an advanced Retrieval-Augmented Generation (RAG) system tailored for wildfire evacuation guidance. It integrates document retrieval from an Elasticsearch index with a fine-tuned GPT-2 model for dynamic instruction generation. The system is deployed via FastAPI, facilitating real-time access to evacuation recommendations through an API interface.

### **WildfireEvacPipeline.py**
The `WildfireEvacPipeline` class implements a structured pipeline for generating wildfire evacuation instructions. It employs a sentence-transformer model for efficient document retrieval and a sequence-to-sequence (seq2seq) model for instruction synthesis. By dynamically retrieving relevant documents from an indexed corpus, this approach enhances the contextual relevance of generated instructions.

### **WildFireEvacRAG.py**
This module defines a two-stage RAG approach for evacuation instruction generation. In the retrieval phase, BM25 scoring is used to extract the most relevant documents from a pre-indexed dataset. The retrieved data, along with user queries, are processed by a fine-tuned BART model in the generation stage, ensuring high accuracy and contextual alignment. The system is designed for command-line execution, offering flexibility in deployment.

---

# **Supporting Modules**

### **RAG.py**
The `WildfireEvacuationRAG` class integrates document retrieval, instruction synthesis, and reinforcement learning. It employs a sentence-transformer model for document retrieval, a GPT-2 model for text generation, and a BERT-based ranking model to optimize instruction quality. The inclusion of reinforcement learning enables continuous adaptation based on user interactions.

### **LlamaIndexQA.py**
This module explores the potential of the LlamaIndex library for generating evacuation instructions. It constructs a document index using the `GPTSimpleVectorIndex` and applies query-based retrieval mechanisms to extract relevant responses. The indexed dataset incorporates critical contextual information, such as terrain characteristics, emergency exits, and past wildfire scenarios, to generate precise evacuation guidance.

### **rl_framework.py**
The `RLAgent` class serves as a reinforcement learning framework for optimizing evacuation instructions. It integrates a pre-trained sentiment analysis model to quantify user feedback and assigns reward values accordingly. Using policy gradient methods, the agent iteratively refines instruction-generation parameters, ensuring that the system evolves toward improved clarity and effectiveness.

---

# **Installation Instructions**

1. Clone the repository:
   ```
   git clone https://github.com/ember.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the necessary datasets and pre-trained models, including:
   - Evacuation zone data (Shapefile/GeoJSON)
   - Fire perimeter data (GeoJSON/KML)
   - Risk assessment and route optimization models
   - Natural language generation models

4. Launch the Streamlit application:
   ```
   streamlit run emberorg.py
   ```

---

# **Usage Guidelines**

1. Access the EMBER system via the Streamlit user interface.
2. Input relevant parameters, including location, fire name, fire year, and the number of evacuees.
3. The system will process the data and generate personalized evacuation instructions based on real-time conditions.
4. Users can interact with the generated instructions, provide feedback, and review visualized evacuation routes.
5. The system continuously incorporates user feedback to enhance instruction generation.

---

# **Contributions**
Contributions to the EMBER project are encouraged. Users can report issues, suggest improvements, or propose new features by opening a GitHub issue or submitting a pull request. The collaborative development approach ensures the system remains robust and adaptable to evolving wildfire response needs.

---

# **License**
This project is distributed under the **MIT License**, allowing for open-source collaboration and continued enhancement of the EMBER system.
