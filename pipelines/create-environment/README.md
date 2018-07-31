# Portable Data Science Pipeline With Docker

In recent years, [Docker](https://www.docker.com/) has emerged in the 
DevOps world as a powerful tool for building versatile reusable 
environments for development and deployment of complex software 
stacks. Docker can be leveraged to run experimentation pipelines in 
isolated and easily recreated environments. In this example we will go 
over a pipeline for a simple data science analysis whose only 
requirement on the host machine is having `docker` installed. 
Instructions on installing Docker can be found at the [official 
website](https://docs.docker.com/install/).

The directory structure for this pipeline is as follows:

```
docker-data-science
├── README.md
├── build-docker-image.sh
├── generate-figures.sh
├── generate-learning-curves.sh
├── validate.sh
├── docker
│   ├── Dockerfile
│   └── requirements.txt
├── results
│   ├── naive_bayes_original.png
│   └── svm_estimator_original.png
└── scripts
    ├── compare-output.py
    ├── generate-figures.py
    └── generate-learning-curves.py
```

There are four stages as well as a directory for the docker image 
definition (`docker/`), past results used to validate re-executions 
(`results/`), and a `scripts/` that include the three python scripts 
for this pipeline. The stages accomplish the following:

  * `build-docker-image`. Builds the docker image, installing and 
    setting up the dependencies specified in the `requirements.txt` 
    file.
  * `generate-learning-curves`. Loads the [Diabetes 
    Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes) and 
    runs two prediction models using distinct methods. At the end, 
    produces CSV files with the learning curve for each method.
  * `generate-figures`. Plots the learning curve for each method.
  * `validate`. Compares the figures for testing bitwise 
    reproducibility.

Using this compartmentalized splitting of stages and keeping 
dependencies inside docker images allow others to easily re-execute 
experimentation pipelines.
