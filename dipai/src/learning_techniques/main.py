import pandas as pd
import os


def check_data_availability(data_directory):
    data_directory = 'src\\learning_techniques\\data'
    for _, _, files in os.walk(data_directory):
        if files: 
            return True
    return False

def main():
    data_available = check_data_availability('src\\learning_techniques\\data')

    if data_available:
        print("Starting data processing...")
        import preprocess
        preprocess.run_preprocess('src\\learning_techniques\\data')
        import train_general
        train_general.main()
        import knowledge_distillation
        knowledge_distillation.run()
        #import meta_learning
        #meta_learning.meta_run()
        print("Data processing completed.")
    else:
        print("No data found. Launching the app with the existing model...")
    import app
    app.start_app()

if __name__ == "__main__":
    main()