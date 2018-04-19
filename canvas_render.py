#!flask/bin/python
from flask import Flask, request, render_template
from pprint import pprint
app = Flask(__name__)

instructor_predictions = None

# def make_public_task(task):
#     new_task = {}
#     for field in task:
#         if field == 'id':
#             new_task['uri'] = url_for('get_task', task_id = task['id'], _external = True)
#         else:
#             new_task[field] = task[field]
#     return new_task

instructor_predictions = [{
             'prev_canvas': '#CANVAS-[{"left":10,"top":10,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":410,"top":10,"width":80,"height":80,"label":"green","shape":"rectangle"},{"left":10,"top":410,"width":80,"height":80,"label":"green","shape":"circle"},{"left":10,"top":110,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":210,"top":210,"width":80,"height":80,"label":"blue","shape":"rectangle"}]',
             'current_instruction': 'add a blue square to bottom-of the blue square on center of the canvas',
             'next_object': {'row': 3, 'col': 2, 'color': 'blue', 'shape': 'square'},
             'ref_obj': {'row': 2, 'col': 2, 'color': 'blue', 'shape': 'square'},
             'final_canvas': '#CANVAS-[{"left":10,"top":10,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":410,"top":10,"width":80,"height":80,"label":"green","shape":"rectangle"},{"left":10,"top":410,"width":80,"height":80,"label":"green","shape":"circle"},{"left":10,"top":110,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":210,"top":210,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":210,"top":310,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":410,"top":410,"width":80,"height":80,"label":"green","shape":"circle"},{"left":310,"top":310,"width":80,"height":80,"label":"green","shape":"circle"},{"left":310,"top":110,"width":80,"height":80,"label":"red","shape":"circle"},{"left":110,"top":310,"width":80,"height":80,"label":"red","shape":"triangle"}]',
             'predicted_instruction': 'add a red triangle to bottom-of the blue square'}, {
             'prev_canvas': '#CANVAS-[{"left":10,"top":410,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":110,"top":410,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":10,"top":10,"width":80,"height":80,"label":"green","shape":"circle"},{"left":210,"top":310,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":410,"top":410,"width":80,"height":80,"label":"green","shape":"rectangle"},{"left":210,"top":210,"width":80,"height":80,"label":"green","shape":"rectangle"},{"left":310,"top":310,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":410,"top":10,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":10,"top":310,"width":80,"height":80,"label":"green","shape":"rectangle"}]',
             'current_instruction': 'add a green square to bottom-right-of the green square on center of the canvas',
             'next_object': {'row': 3, 'col': 1, 'color': 'green', 'shape': 'square'},
             'ref_obj': {'row': 2, 'col': 2, 'color': 'green', 'shape': 'square'},
             'final_canvas': '#CANVAS-[{"left":10,"top":410,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":110,"top":410,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":10,"top":10,"width":80,"height":80,"label":"green","shape":"circle"},{"left":210,"top":310,"width":80,"height":80,"label":"blue","shape":"rectangle"},{"left":410,"top":410,"width":80,"height":80,"label":"green","shape":"rectangle"},{"left":210,"top":210,"width":80,"height":80,"label":"green","shape":"rectangle"},{"left":310,"top":310,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":410,"top":10,"width":80,"height":80,"label":"red","shape":"triangle"},{"left":10,"top":310,"width":80,"height":80,"label":"green","shape":"rectangle"},{"left":110,"top":310,"width":80,"height":80,"label":"green","shape":"rectangle"}]',
             'predicted_instruction': 'add a red triangle to top-of the red triangle'}]

dialog_sample = []


@app.route("/")
def instructor():
    return render_template('instructor_predict.html', data=instructor_predictions)


@app.route("/dialog")
def dialog():
    return render_template('dialog.html', data=dialog_sample)


@app.route('/new_task', methods=['POST'])
def create_task():
    global instructor_predictions
    instructor_predictions = request.json
    return "OK"


@app.route('/new_dialog', methods=['POST'])
def new_dialog():
    global dialog_sample
    dialog_sample = request.json
    return "OK"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
