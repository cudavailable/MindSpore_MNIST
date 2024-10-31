import os
import mindspore
from mindspore import nn
from model import Net
from config import MnistConfig
from data_process import get_dataset
from train import train, test, TrainInfo
from logger import Logger

def forward(data, label, loss_fn, model):
      """ forward propagation """
      logits = model(data)
      loss = loss_fn(logits, label)
      return loss, logits

def main():
      # initialize your model
      model = Net()

      # get loss function
      loss_fn = nn.CrossEntropyLoss()

      # initialize optimizer
      optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=MnistConfig.lr)

      # get gradient function
      grad_fn = mindspore.value_and_grad(forward, None, optimizer.parameters, has_aux=True)

      # get train & test datasets
      train_data, test_data = get_dataset()

      # initialize logger
      log_dir = "./log"
      if log_dir is not None and not os.path.exists(log_dir):
            os.mkdir(log_dir) # create a new directory
      logger = Logger(os.path.join(log_dir, "log.txt"))

      # Set train & test info
      train_info = TrainInfo(model, loss_fn, optimizer, grad_fn, train_data, logger)
      test_info = TrainInfo(model, loss_fn, optimizer, grad_fn, test_data, logger)

      # keep recording best model's parameters
      weights_dir = "./best_weights"
      if weights_dir is not None and not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
      best_weight = {'epoch' : 0, 'metrics' : None, 'correct' : None}

      # train & test
      for t in range(MnistConfig.epoch):
            logger.write(f"Epoch {t + 1}\n-------------------------------\n")
            train(train_info)
            test_loss, correct = test(test_info)

            if best_weight['metrics'] is None or test_loss < best_weight['metrics']:
                  best_weight['epoch'] = t+1
                  best_weight['metrics'] = test_loss # update info
                  best_weight['correct'] = correct
                  model.set_train(False)
                  best_weight_path = os.path.join(weights_dir, "best_model.ckpt")
                  mindspore.save_checkpoint(model, best_weight_path) # save the best model's parameters

      # end of traning
      logger.write("\nTraining completed\n\n")
      logger.write(f"Best Epoch: {best_weight['epoch']}, Accuracy: {(100 * best_weight['correct']):>0.2f}%, Avg loss: {best_weight['metrics']:>8f} \n")
      # close
      logger.close()


      # print("Done!")

if __name__ == "__main__":
      main()
