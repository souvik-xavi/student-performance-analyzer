import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load saved accuracy (if exists)
try:
    with open("accuracy.pkl", "rb") as f:
        accuracy = pickle.load(f)
except FileNotFoundError:
    accuracy = None


def predict_with_model(df):
    """Run predictions using the trained model"""
    X = df[["math score", "reading score", "writing score"]]
    df["Predicted Result"] = model.predict(X)
    return df


def generate_graphs(df):
    """Generate and save graphs for scores + predictions"""
    graph_folder = os.path.join("static", "graphs")
    os.makedirs(graph_folder, exist_ok=True)

    # Clear old graphs
    for old in os.listdir(graph_folder):
        os.remove(os.path.join(graph_folder, old))

    subjects = [
        ("math score", "Math Score"),
        ("reading score", "Reading Score"),
        ("writing score", "Writing Score"),
        ("Predicted Result", "Predicted Result")
    ]

    for col, pretty_name in subjects:
        plt.figure(figsize=(6, 4))
        plt.bar(df.index, df[col])
        plt.xlabel("Students")
        plt.ylabel(pretty_name)
        plt.title(f"{pretty_name} Distribution")
        plt.tight_layout()

        graph_path = os.path.join(graph_folder, f"{col}.png")
        plt.savefig(graph_path)
        plt.close()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Read CSV
    df = pd.read_csv(file)

    # Normalize column names to lowercase internally
    df.rename(
        columns={
            "Math": "math score",
            "Science": "reading score",
            "English": "writing score",
            "Name": "name"
        },
        inplace=True,
    )

    # Predictions
    df = predict_with_model(df)

    # Generate graphs (uses lowercase col names)
    generate_graphs(df)

    # Format columns nicely for display only
    df.rename(columns={
        "name": "Name",
        "math score": "Math Score",
        "reading score": "Reading Score",
        "writing score": "Writing Score",
        "Predicted Result": "Predicted Result"
    }, inplace=True)

    # Collect graph filenames
    graphs = [
        f for f in os.listdir(os.path.join("static", "graphs")) if f.endswith(".png")
    ]

    return render_template(
        "result.html",
        tables=df.to_html(classes="table table-bordered", index=False),
        accuracy=accuracy if accuracy else "N/A",
        graphs=graphs,
    )


if __name__ == "__main__":
    app.run(debug=True)
