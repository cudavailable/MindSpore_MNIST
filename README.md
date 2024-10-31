## MindSpore_MNIST
This is a project about MNIST handwriting recogition with the help of a deep learning framework called MindSpore rather than Pytorch or Tensorflow.

## Config
- epoch = 10  
- lr = 0.001  
- train_batch_size = 64  
- test_batch_size = 64  

## File Description
- MNIST_Data : Training and Testing dataset  
- best_weights : Parameters of the best model (whose test_loss is smallest)  
- config.py : Configurations of the project  
- data_process.py : Some functions about data loading and processing  
- logger.py : Log object  
- model.py : Deep learning neural networks (CNN)  
- train.py : Training and testing process  
- main.py : Main workflow of the project  
