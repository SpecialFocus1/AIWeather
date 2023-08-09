def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def train_model(model, data):
    X = data["weather"]
    y = data["description"]

    y_one_hot = tf.keras.utils.to_categorical(y)

    model.fit(X, y_one_hot, epochs=10)

def predict_weather(model, city):
    data = get_weather_data(city)
    if data:
        weather = data["weather"][0]
        features = [weather["main"]["temp"], weather["wind"]["speed"], weather["humidity"]]
        prediction = model.predict(features)
        description = prediction[0]
        return description
    else:
        return None

def main():
    model = build_model()
    data = get_weather_data("Colorado Springs")
    train_model(model, data)
    description = predict_weather(model, "Colorado Springs")
    print("The weather in Colorado Springs is:", description)

if __name__ == "__main__":
    main()
