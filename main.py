"""
@author: Dr. Ruslan Sherstyukov, March 2026
"""

import os
from datetime import datetime, timedelta
import tensorflow as tf
import IoParametersRecognition as IPR
import IoParametersPostprocessing as IPP


def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (2 * intersection + 1e-7) / (denominator + 1e-7)


def main():
    
    print("Current working directory:", os.getcwd())

    # Define time range
    start_time = datetime(2025, 6, 10, 0, 1)
    end_time   = datetime(2025, 6, 10, 23, 53)
    time_step  = timedelta(minutes=4)

    # Load models
    Models = {
        "ModelTraceF2": tf.keras.models.load_model("F2_trace_IRF.h5", custom_objects={'dice_loss': dice_loss}),
        "ModelTraceF1": tf.keras.models.load_model("F1_trace_IRF.h5", custom_objects={'dice_loss': dice_loss}),
        "ModelTraceE":  tf.keras.models.load_model("E_trace_IRF.h5",  custom_objects={'dice_loss': dice_loss}),
    }

    # Form Io parameters DB
    current_time = start_time
    while current_time <= end_time:
        IPR.IonogramDatabase(
            Models,
            year=current_time.strftime("%Y"),
            month=current_time.strftime("%m"),
            day=current_time.strftime("%d"),
            hour=current_time.strftime("%H"),
            minute=current_time.strftime("%M")
        )
        print(current_time.strftime("%Y-%m-%d %H:%M"))
        current_time += time_step

    # DB Postprocessing
    IPP.ParametersPostprocessing(
        year=start_time.strftime("%Y"),
        month=start_time.strftime("%m"),
        day=start_time.strftime("%d")
    )


if __name__ == "__main__":
    main()
