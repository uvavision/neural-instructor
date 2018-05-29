# Interactive Neural Bot Drawing

## Overview

In this project we designed two neural network based models that could interact with each other to finish a simple 2D shape painting. The painting canvas is 5x5 grid where shapes (circle, square, rectangle) of different colors (red, green, blue) could be placed on. Given a goal canvas and an empty scratch canvas, the two agents interact with each other to update the scratch canvas until the scratch canvas is the same with the goal canvas. At each step, the instructor model takes the current scratch canvas and the goal canvas as input, and generates a natural language instruction for the painter model to follow. The painter model takes the generated instruction as input, and updates the current scratch canvas according to the instruction. This interaction continues until the current scratch canvas is the same with the goal canvas.

In correspondence with real-world scenario, we assume noises in both models. The instructor model could issue incorrect instructions, and the painter model could incorrectly execute correct instructions. To make  the interaction fault tolerant , the instructor model is designed to issue two kinds of instructions: *add instructions* and *delete instructions*. 

## The Instructor Bot

### Instruction Format

The instructor model is trained to generate instructions of two action types: Add instruction and Delete instruction. Instructions could use absolute or relative referencing when referring a location. There are 9 locations for absolute referring: `top-left, top-right, bottom-left, bottom-right, center, top-middle, bottom-middle, left-middle, right-middle`. There are 8 offset location for use when using relative referencing: `top-of, bottom-of, right-of, left-of, top-left-of, top-right-of, bottom-left-of, bottom-right-of`. Some example instructions: 

- Add instructions: 
  - `Add one green triangle one at top-left of the canvas`
  - `Now place a red triangle one at left-of of the blue triangle object.`
  - `Now place a blue triangle one at top-of of the one at bottom-right of the canvas.`
- Delete instructions: 
  - `Now get rid of the one at right-of of the green circle one at top-left of the canvas.`
  - `Delete the one at left-middle of the canvas.`
  - `Please remove the green circle object at bottom-left-of of the green square one.`

### The Instructor Model

The instructor model is a recurrent neural network language model. The design is largely based on the "top-down image captioning model" proposed in the paper [Bottom-Up and Top-Down Attention for Image Captioning and Viusal Question Answering](https://arxiv.org/pdf/1707.07998.pdf). Attention tensor $R^{5\times 5\times 4}$ is computed from the current scratch canvas and the goal canvas. The attention tensor has 4 channels: the first the color channel, the second the shape channel, the third a flag channel indicating whether an object is present in the current scratch canvas, and the last a flag channel indicating whether an object is present in the goal canvas. The attention tensor is flattened as 25 attention vectors of dimension 4 to be used in the model. Check the file `instructor_new.py` for details.

![topdown model](https://s3.amazonaws.com/github-share/topdown.PNG "Top Down Image Captioning Model")

## The Painter Bot

### Message Format

We assume messages issued by the instructor has multiple instructions. Instructions could be Add or Delete instructions, could be of absolutely or relatively referenced instructions.  One example message could be

```python
Now place a red circle one at bottom-of the blue circle one; Add a blue circle at top-left of the canvas; Now get rid of the one at left-of of the triangle one.
```

The message has three instructions: the first an Add instruction with relative location referencing , the second an Add instruction with absolute location referencing, and the last Delete instruction with relative location referencing.

### The Painter Model

#### Recurrent Painter Model

The painter model takes a message and the canvas before the message as inputs, and predicts an updated canvas based on instructions in the message. Although the painter model is designed to process multiple instructions, internally it processes instructions one-by-one: at each step it takes an instruction and previously predicted canvas as input, and constructs an intermediate canvas; intermediate canvas and the next instruction are again input to the next step.  Figure \ref{painter_model} presents a schematic illustration of the painter model.

![A schematic illustration of the painter model\label{painter_model}](https://s3.amazonaws.com/github-share/painter_model.PNG)

#### Painter Model in One Step

At each step the painter model takes an instruction and a canvas as input, and predicts a command that is parameterized by three terms: action, object and location. For example, in the instruction `Now place a blue circle at top-left of the canvas` , the action is "add", the object is a "blue circle", and the location is "top-left" (row=0, col=1). However, if the instruction is a Delete instruction, the model only needs to predict the location of the target object. 

We use a LSTM network to encode input instruction $h_t = \text{LSTM}(h_{t-1}, x_t; W_1),$where $x_t$ is the word embedding of the $t$th word.  We use the last hidden state vector $h_T$ as the encoded representation of the input instruction.  The action (add or delete) and target object (color and shape) are  directly predicted from the instruction encoding using linear units:

- Action: $P(a|h_T) = \text{softmax}(W_a h_T + b_a)$
- Color: $P(c|h_T) = \text{softmax}(W_c h_T + b_c)$
- Shape: $P(s|h_T) = \text{softmax}(W_s h_T + b_s)$

 When predicting the location of the target object, we consider two different types of instructions:

- Absolute location referencing instruction: the location of target object is directly predicted from the instruction encoding $P(l|h_T) = \text{softmax}(W_l h_T + b_l)$
- Relative location referencing instruction: the model has to first retrieve the referencing object from the input canvas, and use the location of referencing object to compute the location of the target object. The idea is similar to [Memory Networks](https://arxiv.org/abs/1410.3916) in that we use the canvas as the memory of the network. In our  case we use the *hard attention* algorithm to retrieve just one referencing object. Once the referencing object is retrieved, the location of the target object is computed by adding the reference object location and an offset predicted from the instruction encoding. 

#### Painter Model Training

The painter model take an message and the canvas before the message as input, and predicts the canvas after the message. In such a case, the only supervision for training the painter model is the canvas *before* and *after* the message. We don't have the canvas sate after *each* instruction, we only the canvas state after *a series* of instructions. The painter model is trained using reinforcement learning. We designed different reward functions for different action types of instruction. Note that the model has to predict instruction action type as well. 

- Reward function for Add instructions

![Reward function for add instructions](https://s3.amazonaws.com/github-share/step_reward_model_add.PNG)

- Reward function for Delete instructions

![Reward function for delete instructions](https://s3.amazonaws.com/github-share/step_reward_model_delete.PNG)

### Dataset

| Dataset    | Number of Messages | Message Format                                               |
| ---------- | :----------------- | ------------------------------------------------------------ |
| Training   | 43K                | Messages have exactly 3 instructions.                        |
| Validation | 3K                | Messages have variable number of instructions, max number = 12. |

Use the script `dial_core_sampling.py` to generate data; setting the flag `generate_dialog_slice_data` inside the script to generate training or validation data.

### Performance

We measured the performances of two models. 

- Using GT ref type: assume the knowledge of instruction reference type (absolute reference or relative reference) 
- Predict ref type: The model needs to predict instruction reference type. 

All models use an Adam optimizer with a fixed learning rate of 5e-4, use batch size 32, trained for 200 epochs.

We use *success rate*  to measure the performances of different models. A model takes a message and the canvas before the message as input, and predicts the canvas after the message. If the predicted one matched with the ground-truth one, it’s a success. 

| Model         | Training Set Success Rate | Validation Set Success Rate |
| ------------- | ------------------------- | --------------------------- |
| GT ref type   | 99.9%                     | 99.3%                       |
| Pred ref type | 68.5%                     | 39.3%                       |

