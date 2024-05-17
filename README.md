# MHealth_text_classification

## Project Overview
`MHealth_text_classification` is an AI-based triage system. This system is designed to analyze text communications in telehealth platforms to assess mental health statuses, leveraging Natural Language Processing (NLP) and machine learning techniques. The primary objective is to improve the identification and management of mental health issues through text-based telehealth services.


## Author
Christian Bernard Ochieng

## Features
- Text classification using Natural Language Processing (NLP).
- Machine learning models to detect patterns indicative of mental health conditions like depression and anxiety.
- Implementation of multiple models including Multinomial Naive Bayes, Fully Connected Neural Networks, LSTM, and Convolutional Neural Networks.

## Installation
To set up this project locally, follow these steps:
1. Clone the repository:
2. Navigate to the project directory:
3. Install the required dependencies:


## Usage
To run the text classification model:
1. Ensure you have the necessary datasets loaded as described in the Data Collection section of the project.
2. Execute the main script:


## Data
The project uses anonymized text messages from a telehealth platform, processed to remove identifiers and ensure compliance with GDPR and HIPAA regulations.

## Configuration
Adjust model parameters and dataset paths in the `config.py` file as needed to optimize performance or adapt to new data sources.

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request with your updates.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- LVCT Health for providing the platform and dataset.
- Mentors and colleagues from the Masters of Data Science and Artificial Intelligence program.

## Citation
Please cite this project in your publications if it helps your research. The following is a bibtex reference:

@misc{ochieng2024mhealth_text_classification,
title={Development of an AI-Based Triage System for Mental Health Assessment in Telehealth Text Communications},
author={Christian Bernard Ochieng},
year={2024},
publisher={GitHub},
journal={GitHub repository},
howpublished={\url{https://github.com/christianoben/DSAI/tree/main}}
}