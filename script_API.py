# Imports
from fastapi import FastAPI, Path
import pickle
from pydantic import BaseModel
import json

app = FastAPI()

# get model previously developped
model = pickle.load(open("model", 'rb'))
scaler = pickle.load(open("scaler_MinMax", 'rb'))

#Define basemodel player. the name is kept here in case it would be needed for other purpose than prediction
class Player(BaseModel):
    name:str
    GP:int
    MIN:float
    PTS:float
    FGM:float
    FGA:float
    FGp:float
    threePointsMade:float
    threePointsAttempts:float
    threePointsPercent:float
    FTM:float
    FTA:float
    FTp:float
    OREB:float
    DREB:float
    REB:float
    AST:float
    STL:float
    BLK:float
    TOV:float
        

# One single post request
@app.post("/predict-investment")
async def return_prediction(player: Player):
  
    # need to convert player to 2D array to make it useable by the model
    player_array = [[
                    player.GP,
                    player.MIN,
                    player.PTS,
                    player.FGM,
                    player.FGA,
                    player.FGp,
                    player.threePointsMade,
                    player.threePointsAttempts,
                    player.threePointsPercent,
                    player.FTM,
                    player.FTA,
                    player.FTp,
                    player.OREB,
                    player.DREB,
                    player.REB,
                    player.AST,
                    player.STL,
                    player.BLK,
                    player.TOV 
                    ]]
    
    player_scaled = scaler.transform(player_array)
    
    result = model.predict(player_scaled)
    
    
    # need to convert array to list to jsonify it
    resultAsList = result.tolist()

    return json.dumps(resultAsList)


