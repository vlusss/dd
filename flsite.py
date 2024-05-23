import pickle

import numpy as np
from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"}
        ]

loaded_model_knn = pickle.load(open('model/Iris_pickle_file_knn', 'rb'))
loaded_model_linel = pickle.load(open('model/Iris_pickle_file_jilie', 'rb'))
loaded_model_logic = pickle.load(open('model/Iris_pickle_file_logic', 'rb'))
loaded_model_tree = pickle.load(open('model/Iris_pickle_file_tree', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Петрачков Александр ПрИ-201",
                           menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это " + pred[0])


@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Линейная регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1'])]])
        pred = loaded_model_linel.predict(X_new)
        return render_template('lab2.html', title="Линейная регрессия", menu=menu,
                               class_model=pred[0][0])


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    dct = {
        0: "blue",
        2: "green",
        1: "white"
    }
    from matrix import get_matrix_beloved_color

    get_matrix_beloved_color(loaded_model_logic, 'logic')
    if request.method == 'GET':
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_logic.predict(X_new)
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu,
                               class_model="Любимый цвет: " + dct[pred[0]])


@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    from matrix import get_matrix_beloved_color

    get_matrix_beloved_color(loaded_model_tree, 'tree')

    if request.method == 'GET':
        return render_template('lab4.html', title="Дерево решений", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_tree.predict(X_new)
        return render_template('lab4.html', title="Дерево решений", menu=menu,
                               class_model="Любимый цвет: " + pred[0])


@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('list1')),
                       float(request.args.get('list2')),
                       float(request.args.get('list3')),
                       float(request.args.get('list4'))]])
    pred = loaded_model_tree.predict(X_new)

    return jsonify(color=pred[0])


@app.route('/api_v2', methods=['get'])
def get_sort_v2():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['list1']),
                       float(request_data['list2']),
                       float(request_data['list3']),
                       float(request_data['list4'])]])
    pred = loaded_model_tree.predict(X_new)

    return jsonify(color=pred[0])


if __name__ == "__main__":
    app.run(debug=True)
