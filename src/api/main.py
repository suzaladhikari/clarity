from fastapi import FastAPI 

app = FastAPI(title = "API for Clarity", description= "API for stock volatility prediction", version = "1.0.0")


## Creating the elements 
@app.get('/hello')
def say_hello():
    return "Hey There"