#!flask/bin/python
from flask import Flask, request, render_template
from pprint import pprint
from Shape2DInstructorData import render_canvas
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

instructor_predictions = []
dialog_sample = []
new_instructor_predictions = []


@app.route("/")
def instructor():
    return render_template('instructor_predict.html', data=instructor_predictions)


@app.route("/instruction_prediction")
def show_instruction():
    return render_template('instructor_predict_new.html', data=new_instructor_predictions)


@app.route("/dialog")
def dialog():
    return render_template('dialog.html', data=dialog_sample)


@app.route('/new_task', methods=['POST'])
def create_task():
    global instructor_predictions
    instructor_predictions = request.json
    return "OK"

@app.route('/new_instruction', methods=['POST'])
def add_new_instruction():
    global new_instructor_predictions
    new_instructor_predictions = request.json
    return "OK"

@app.route('/new_dialog', methods=['POST'])
def new_dialog():
    global dialog_sample
    dialog_sample = request.json
    for sample in dialog_sample:
        sample['prev_canvas'] = render_canvas(sample['prev_canvas'])
        sample['next_canvas'] = render_canvas(sample['next_canvas'])
        sample['final_canvas'] = render_canvas(sample['final_canvas'])
    return "OK"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
