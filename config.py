import os
from dotenv import load_dotenv

# GET API KEY:
load_dotenv("alpha.env")
gl_key = os.getenv("API_KEY")

#  ================ CONFIG =====================
config = {
    "alpha_vantage": {
        "key": gl_key,
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    }, 
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 30,
        "dropout": 0.7,
    },
    "model_original": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}
#  ================ ^^ CONFIG ^^ =====================

def logical(user_input):
  """Converts user input to a boolean value, handling variations of 'yes' and 'no'.

  Args:
      user_input (str): The user's input string.

  Returns:
      bool: True if the input represents 'yes' (case-insensitive), False otherwise.

  Raises:
      ValueError: If the input is not recognized as a valid yes/no variation.
  """

  valid_yes = ("y", "yes")
  valid_no = ("n", "no")

  user_input_lower = user_input.lower()  # Convert to lowercase for case-insensitivity

  if user_input_lower in valid_yes:
    return True
  elif user_input_lower in valid_no:
    return False
  else:
    raise ValueError(f"Invalid input: '{user_input}'. \nPlease enter 'y'/'yes' or 'n'/'no'.")