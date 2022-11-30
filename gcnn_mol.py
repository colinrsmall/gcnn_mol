from utils.args import TrainArgs
import train

if __name__ == "__main__":
    train_args = TrainArgs().parse_args()
    train.train_model(train_args)
