Build docker image
docker build -t image_name .

Run container
docker run --name container_name -p 8000:8000 image_name

Output will contain
INFO: Uvicorn running on http://0.0.0.0:8000

Use this url in chrome to see the model frontend; use http://0.0.0.0:8000/docs for testing the model in the web interface.

Query model
Via web interface (chrome):
  http://0.0.0.0:8000/docs -> test model
Via python client:
  client.py
Via curl request:
    curl -X POST "http://0.0.0.0:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'