import PySimpleGUI as sg
from learning import *


def handle_learn(img: list[list[int]], perceptrons: list[Perceptron]) -> list[int]:
    image = get_image_vector_from_list(img)
    results = []
    for perceptron in perceptrons:
        if predict(image, perceptron) == 1:
            results.append(perceptron.number)
    return results
    

setup_logger()
train_images = get_training_images()
perceptrons = get_perceptrons(train_images)

learning_rate = 0.1


layout = []

for i in range(6):
    row = [sg.Button("", size=(6, 3), key=(i, j), button_color=("white", "white"), pad=(0, 0)) for j in range(5)]
    layout.append(row)

button_column = sg.Column(
    [
        [
            sg.Button("Learn", key="learn", size=(6, 2)),
            sg.Button("Predict", key="predict", size=(6, 2)),
            sg.Button("Clear", key="clear", size=(6, 2)),
        ],
        [sg.Text("", key="info_label", size=(15, 1))],
    ],
    element_justification="center",
)

layout.append([button_column])

window = sg.Window("Simple", layout, element_justification="center", size=(400, 500))

pixel_states = [[-1 for _ in range(5)] for _ in range(6)] 

while True:
    event, values = window.read()  # type: ignore
    if event == sg.WIN_CLOSED:
        break
    if isinstance(event, tuple):
        i, j = event
        button = window[event]
        current_state = pixel_states[i][j]
        new_state = -current_state
        pixel_states[i][j] = new_state
        new_color = "black" if new_state == 1 else "white"
        button.update(button_color=(new_color, new_color))
    elif event == "learn":
        train_all_perceptrons(get_training_images(), perceptrons, learning_rate)
        window["info_label"].update("Learned")  # type: ignore
    elif event == "predict":
        window["info_label"].update(handle_learn(pixel_states, perceptrons))  # type: ignore
    elif event == "clear":
        window["info_label"].update("")  # type: ignore
        for i in range(6): 
            for j in range(5):
                layout[i][j].update(button_color=("white", "white"))
                pixel_states[i][j] = -1 

window.close()
