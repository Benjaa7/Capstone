# Instancias

- AM_large.xlsx
- PM_small.xlsx

Contiene hojas:

- Passengers: pasageros a transporar, indican sus coordenadas de origen, de destino y prioridad. Para el caso AM se indica la hora desde la cual se le puede recoger y la hora de inicio de su turno. Para el caso PM se indica el fin de su turno.
- Vehicles: tipos de vehículos, con sus cantidades disponibles, capacidades y costo por uso.
- Union constraints: restricciones sindicales, para cada nivel de prioridad se indica el tiempo máximo que puede esperar y el tiempo máximo en vehículo. Para el caso AM el tiempo de espera es la antelación al comienzo de su turno. Para el caso PM el tiempo de espera es desde el fin de su turno.
- Others: datos varios como fecha en la cual se ejecutarán los viajes, costo por distancia, tiempos de parada.

# Matrices de tiempos y distancia

- Archivos con nombres en formato `times_{INSTANCIA}_{hora}.csv`

Contiene columnas:

- origin: coordenadas de origen
- destination: coordenadas de destino
- time: tiempo pronosticado de viaje en segundos
- distance: distancia pronosticada de viaje en metros

# Muestra de tiempos reales

- real_times_sample.csv

Contiene columnas:

- orig_latitude: coordenadas de origen
- orig_longitude: coordenadas de origen
- dest_latitude: coordenadas de destino
- dest_longitude: coordenadas de destino
- departure_at: fecha y hora en que empezó el viaje
- predicted_time: tiempo pronosticado de viaje
- real_time: tiempo real de viaje observadoss
