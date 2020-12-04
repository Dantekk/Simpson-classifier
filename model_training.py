from classifier import Classifier

def model_testing(c, test_generator):
    c.model_load(path_model="simpson_classifier.h5")
    metrics_test = c.model_evaluate(test_generator)
    print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

    # Final model have these metrics on test set :
    # Test Accuracy = 0.9470 - Test Loss = 0.2402

def model_training(c, train_generator, validation_generator):
    c.configure_model()
    c.model_print_summary()

    c.model_training(train_generator, validation_generator, epochs_number=100)

    metrics_train = c.model_evaluate(train_generator)
    metrics_validation = c.model_evaluate(validation_generator)
    print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
    print("Validation Accuracy = %.4f - Validation Loss = %.4f" % (metrics_validation[1], metrics_validation[0]))

    # Train Accuracy = 0.9581 - Train Loss = 0.1521
    # Validation Accuracy = 0.9431 - Train Loss = 0.2307

    c.model_save('simpson_classifier')

def main():
    path_dataset = './dataset/'
    c = Classifier()

    # I have used 18136 images :
    # 12688 images for train set.
    # 3619  images for validation set.
    # 1829  images for test set.
    train_generator, validation_generator, test_generator = c.pre_processing(
            path_dataset = path_dataset,
            batch_size = 128,
            image_size = (200, 200),
            class_mode = 'categorical')

    testing_set_mode = True

    # After model selection, you can calculate accuracy and loss on test set generator
    if testing_set_mode :
        model_testing(c, test_generator)
    else :
        model_training(c, train_generator, validation_generator) # Model training


if __name__ == "__main__":
    main()
