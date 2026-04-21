import csv
import matplotlib.pyplot as plt

def plot_learning_curve(log_path="training_log.csv", output_path=None):
    steps = []
    train_losses = []
    # Read the CSV log file
    with open(log_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse epoch and training loss
            if row.get("global_step"):
                steps.append(int(row["global_step"]))
                train_losses.append(float(row["train_loss"]))
    # Plot the training loss curve
    plt.figure()
    plt.plot(steps, train_losses, marker='o', label="Training Loss")
    plt.title("Training Loss over Steps")
    plt.xlabel("Global Step")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_validation_loss(log_path="training_log.csv", output_path=None):
    # Similar implementation: read epochs and val_loss, then plot
    steps = []
    val_losses = []
    with open(log_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Only include epochs where validation loss is available (non-empty)
            if row.get("val_loss") not in (None, "", "NA"):
                steps.append(int(row["global_step"]))
                val_losses.append(float(row["val_loss"]))
    plt.figure()
    plt.plot(steps, val_losses, marker='o', color='orange', label="Validation Loss")
    plt.title("Validation Loss over Steps")
    plt.xlabel("Global_step")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_perplexity(log_path="training_log.csv", output_path=None):
    steps = []
    perplexities = []
    with open(log_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("perplexity") not in (None, "", "NA"):
                steps.append(int(row["global_step"]))
                perplexities.append(float(row["perplexity"]))
    plt.figure()
    plt.plot(steps, perplexities, marker='o', color='green', label="Perplexity")
    plt.title("Perplexity over Steps")
    plt.xlabel("Global Step")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    log_file_path = "output/thePile_eLM_InD/training_log.csv"
    plot_learning_curve(log_file_path, output_path="learning_curve.png")
    plot_validation_loss(log_file_path, output_path="validation_curve.png")
    plot_perplexity(log_file_path, output_path="perplexity.png")
