FROM python:3.9
WORKDIR /app
COPY requirements.pip ./requirements.pip
RUN pip3 install -r requirements.pip
EXPOSE 8501
COPY names.txt /app/names.txt
COPY app.py /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
