import sys
import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        self.num_classes = num_classes
        self.num_filters = 32
        out_units = 4*4 * self.num_filters*4
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ), # Out: [32, 32]
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ), # Out: [16, 16]
            
            # Layer 2
            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=self.num_filters*2,
                kernel_size=5,
                stride=1,
                padding=2
            ), # Out: [16, 16]
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ), # Out: [8, 8]

            # Layer 3
            nn.Conv2d(
                in_channels=self.num_filters*2,
                out_channels=self.num_filters*4,
                kernel_size=5,
                stride=1,
                padding=2
            ), # Out: [8, 8]
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ) # Out: [4, 4]
        )
        

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16] 4,4?
        #self.num_output_features = 32 * 32 * 32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # Layer 1
            nn.Linear(
                in_features=out_units,
                out_features=64
            ),
            nn.ReLU(),

            # Layer 2
            nn.Linear(
                in_features=64,
                out_features=10
            )
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        CNN_out = self.feature_extractor(x)
        out = self.classifier(CNN_out)

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 0.05
    early_stop_count = 4
    opt = "SGD"
    weight_decay = 0.000
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        opt = opt,
        weight_decay = weight_decay
    )
    trainer.train()
    create_plots(trainer, "task2a")

    trainer.model.eval()

    # Train accuracy
    train_loss, train_acc = compute_loss_and_accuracy(
        trainer.dataloader_train, trainer.model, trainer.loss_criterion
    )
    print("Train loss: ", train_loss)
    print("Train accuracy: ", train_acc)

    # Validation accuracy
    val_loss, val_acc = compute_loss_and_accuracy(
        trainer.dataloader_val, trainer.model, trainer.loss_criterion
    )
    print("Validation loss: ", val_loss)
    print("Validation accuracy: ", val_acc)

    # Test accuracy
    test_loss, test_acc = compute_loss_and_accuracy(
        trainer.dataloader_test, trainer.model, trainer.loss_criterion
    )
    print("Test loss: ", test_loss)
    print("Test accuracy: ", test_acc)



if __name__ == "__main__":
    main()
