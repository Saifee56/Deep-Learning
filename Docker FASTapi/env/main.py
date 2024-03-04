from fastapi import FastAPI

app=FastAPI()
@app.get("/")
def index():
    return {"Details":"Hello, I am learning Docker and Kubernetes"}
@app.get("/my_name")
def my_name():
    return {"I am Safuan Alam Saifee. New to docker"}
