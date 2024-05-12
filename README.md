## Futtatás

main.py futtatása

### Szükséges csomagok

- Pandas
- NumPy

## Használat

### Példányosítás
A model példányosítása során megadható, a vágási kritérium. Jelenleg két metódus támogatott:

- MSE
- MAE

```python

# MSE használata
model = Node(method="MSE")

# MAE használata
model = Node(method="MAE")

```

### Adathalmaz megadása
A model számára a felhasználandó tanító, illetve teszt adatokat tartalmazó csv file-t, a datasets mappában kell elhelyezni, az elérését pedig a data változóban lehet megadni.

### Adathalmaz szeparálása
A model képes teszt és training halmazra szétbontani a megkapott csv file tartalmát, melyért a model split_data metódusa felel.

Bemeneti paraméterei rendre:

- az importált adatkészlet
- random_state: seed, mindig azonosan bontja az adathalmazt ugyan azon seed megadása esetén
- ratio: meghatározza a teszt adatok arányát a beadott dataset-ben.

```python

train_split, test_split = model.split_data(data, random_state=42, ratio=0.01)

```

### Modell tanítása

Felépíti a döntési fát a megadott X adatkészlet és Y célértékek alapján.


```python

model.fit(X=X_train, Y=Y_train['quality'].values.tolist())

```

### Modell kiiratása

```python

model.print()

```

### Regressziós érték megjósolása

A beadott listányi X-ekre megjósolja a felépített fa alapján a hozzájuk tartozó Y-okat.

```python

X_test['yhat'] = model.predict(X_test)

```

