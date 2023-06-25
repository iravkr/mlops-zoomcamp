FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter_cli.py", "model.bin", "./"]

# Add the year and month as build arguments
ARG YEAR
ARG MONTH

# Set the environment variables for year and month
ENV YEAR=${YEAR}
ENV MONTH=${MONTH}

ENTRYPOINT ["pipenv", "run", "python", "starter_cli.py", "--year", "$YEAR", "--month", "$MONTH"]

# Install pandas library

RUN pipenv run pip install pandas scikit-learn==1.2.2 pyarrow



# ENTRYPOINT ["starter_cli:app"]  
