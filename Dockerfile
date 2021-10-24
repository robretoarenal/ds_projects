#Create a base image.
FROM continuumio/miniconda3

#Set work directory
WORKDIR /usr/src

#Update pip
RUN pip install --upgrade pip

#Install dependencies using a requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY environment.yml environment2.yml .
RUN conda env create -f environment.yml
RUN conda env create -f environment2.yml

COPY . .
