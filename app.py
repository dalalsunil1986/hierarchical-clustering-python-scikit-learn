from flask import Flask, render_template
from agglomerative_clusterer import cluster_data
import numpy as np
import json

app = Flask(__name__)

@app.route("/")
def home():
    cluster_1, cluster_2 = cluster_data()
    conv_cluster_1_x = json.dumps(np.array(cluster_1)[:, 0].tolist())
    conv_cluster_1_y = json.dumps(np.array(cluster_1)[:, 1].tolist())

    conv_cluster_2_x = json.dumps(np.array(cluster_2)[:, 0].tolist())
    conv_cluster_2_y = json.dumps(np.array(cluster_2)[:, 1].tolist())
    return render_template("index.html",
                           cluster_1_x=conv_cluster_1_x,
                           cluster_1_y=conv_cluster_1_y,
                           cluster_2_x=conv_cluster_2_x,
                           cluster_2_y=conv_cluster_2_y)


if __name__ == "__main__":
    app.run(debug=True)