# Week 4

This directory contains the scripts and artifacts for the homework of module [`04-deployment`](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/cohorts/2024/04-deployment/homework.md).

The main goal of the assignment is to deploy the ride duration model for batch inference.

The inference script [predict.py](predict.py) can be executed either locally, using a virtual environment, or inside a Docker container.

## Local mode

#### Poetry env

To run the inference script locally, first you need to install Poetry. This repository uses Poetry for managing your Python dependencies. You can follow these steps to build a virtual environment:

1. Install Poetry by following the instructions in the [official documentation](https://python-poetry.org/docs/#installation).
2. Navigate to the `Week_4` directory.
3. Run the following command to create a virtual environment and install the required dependencies:
    ```
    poetry install
    ```
4. Activate the virtual environment:
    ```
    poetry shell
    ```

#### Run the inference script locally

Once you have installed and activated the Poetry virtual environment, you can run the inference script for a specific year and month using the following command:
```
python predict.py 2023 5
```

## Docker

You can also run the inference script in a Docker container.

#### Build the Docker image

To set up the environment for running the prediction pipeline, follow these steps:

1. Install Docker on your machine.
2. Clone the repository to your local machine.
3. Navigate to the `Week_4` directory.
4. Build the Docker image using the provided Dockerfile:
    ```
    docker build -t prediction-pipeline .
    ```

#### Run the script using Docker

To run the inference script inside the Docker container, you can execute the following command:
```
docker run -it prediction-pipeline
```
By default, the inference script inside the Docker container is executed for the data of March 2023. To run the inference script for a different year/month, you can provide your values as arguments to the Python script:
```
docker run -it prediction-pipeline 2023 5
```
