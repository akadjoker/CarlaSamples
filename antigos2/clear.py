import carla
import cv2
import numpy as np
import time
import random
import sys
import os
import datetime
import math
import argparse

 

def setup_weather_and_time(world, hour=17, weather="Clear"):
    """
    Configura o clima e a hora do dia no CARLA.
    
    Parâmetros:
    - world: objeto mundo do CARLA
    - hour: hora do dia (0-23)
    - weather: tipo de clima ("Clear", "Cloudy", "Rain", "HardRain", "Sunset")
    """
    # Configura a hora do dia
    weather_params = world.get_weather()
    
    # Define a hora do dia (azimute solar)
    # 0 = meia-noite, 90 = meio-dia, 180 = meia-noite, 270 = meio-dia
    if hour >= 0 and hour < 12:
        # Manhã: 0h = 180°, 12h = 90°
        azimuth = 180 - (hour * 7.5)
    else:
        # Tarde/Noite: 12h = 90°, 24h = 0°
        azimuth = 90 - ((hour - 12) * 7.5)
    
    # Calcula a elevação solar (altura no céu)
    # Ao meio-dia a elevação é máxima, à meia-noite é mínima
    if hour >= 6 and hour <= 18:
        # Dia: 6h = 0°, 12h = 90°, 18h = 0°
        if hour <= 12:
            elevation = (hour - 6) * 15
        else:
            elevation = (18 - hour) * 15
    else:
        # Noite: elevação negativa (sol abaixo do horizonte)
        if hour > 18:
            elevation = -15
        else:
            elevation = -15
    
    weather_params.sun_azimuth_angle = azimuth
    weather_params.sun_altitude_angle = elevation
    
    # Configura o tipo de clima
    if weather == "Clear":
        weather_params.cloudiness = 10.0
        weather_params.precipitation = 0.0
        weather_params.precipitation_deposits = 0.0
        weather_params.wetness = 0.0
    elif weather == "Cloudy":
        weather_params.cloudiness = 80.0
        weather_params.precipitation = 0.0
        weather_params.precipitation_deposits = 0.0
        weather_params.wetness = 0.0
    elif weather == "Rain":
        weather_params.cloudiness = 80.0
        weather_params.precipitation = 60.0
        weather_params.precipitation_deposits = 40.0
        weather_params.wetness = 40.0
    elif weather == "HardRain":
        weather_params.cloudiness = 90.0
        weather_params.precipitation = 90.0
        weather_params.precipitation_deposits = 80.0
        weather_params.wetness = 80.0
    elif weather == "Sunset":
        weather_params.cloudiness = 15.0
        weather_params.precipitation = 0.0
        weather_params.precipitation_deposits = 0.0
        weather_params.wetness = 0.0
        weather_params.sun_azimuth_angle = 90.0
        weather_params.sun_altitude_angle = 15.0  # Sol baixo no horizonte
    
    # Para final de tarde (17h),  
    # aumenta a densidade de neblina e a distância da neblina
    if hour >= 17 and hour <= 19:
        weather_params.fog_density = 10.0
        weather_params.fog_distance = 75.0
        weather_params.fog_falloff = 1.0
        
    world.set_weather(weather_params)
    print(f"Clima configurado: {weather}, Hora: {hour}:00")

def main():



    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
 
    world = client.get_world()

    

    world.unload_map_layer(carla.MapLayer.All)

    # world.unload_map_layer(carla.MapLayer.Decals)
    # world.unload_map_layer(carla.MapLayer.Props)
    # world.unload_map_layer(carla.MapLayer.StreetLights)
    # world.unload_map_layer(carla.MapLayer.Foliage)
    # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    # world.unload_map_layer(carla.MapLayer.Particles)
    # world.unload_map_layer(carla.MapLayer.Walls)
    # world.unload_map_layer(carla.MapLayer.Buildings)
 
    setup_weather_and_time(world)

 

    
 
   
        
 
        
    
    

if __name__ == '__main__':
 
        main()
 
